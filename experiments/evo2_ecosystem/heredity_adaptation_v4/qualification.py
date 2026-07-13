"""Prospectively qualify the first harmful, viable r4 actuation-cost shock."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from microcosmos.heredity import (
    R4_CLONE,
    R4_STANDARD_PARAMETRIC,
    fixed_r4_policy,
    mutate_cppn_r4,
)

from ..episode import SimulatorConfig, evaluate_manifest, simulator_source_sha256
from ..founder_artifacts import founder_index_sha256, load_founder_index
from ..protocol import EventKind, manifest_sha256
from .manifest_generator import _manifest, _founders


MULTIPLIER_GRID = (1.25, 1.50, 2.00, 3.00)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 74_219


def uniform_exploration_policy(parent, parent_stats, population_stats, context):
    """Trusted qualification-only policy that exercises every r4 action."""
    del parent_stats, population_stats
    return mutate_cppn_r4(parent, jnp.zeros(6, dtype=jnp.float32), context)


def _shock_episodes(manifest, evaluation):
    return tuple(
        episode
        for world, episode in zip(manifest.worlds, evaluation.episodes, strict=True)
        if world.event_kind is EventKind.ACTUATION_COST_SHIFT
    )


def _pair_effects(manifest, evaluation):
    grouped = defaultdict(dict)
    founder = {}
    for world, episode in zip(manifest.worlds, evaluation.episodes, strict=True):
        grouped[world.pair_id][world.event_kind] = float(episode.primary_score)
        founder[world.pair_id] = world.founder_id
    return [
        (founder[pair_id], scores[EventKind.ACTUATION_COST_SHIFT] - scores[EventKind.NULL])
        for pair_id, scores in grouped.items()
    ]


def _founder_bootstrap_upper(effects) -> float:
    grouped = defaultdict(list)
    for founder_id, effect in effects:
        grouped[founder_id].append(effect)
    founder_means = np.asarray([np.mean(values) for values in grouped.values()])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(founder_means, size=(BOOTSTRAP_REPLICATES, len(founder_means)), replace=True)
    return float(np.quantile(np.mean(samples, axis=1), 0.975))


def _generation_gain(episode) -> int:
    population = episode.shock_population
    if population is None:
        raise RuntimeError("qualification requires captured shock populations")
    pre = int(np.max(np.where(np.asarray(population.alive), np.asarray(population.generation), 0)))
    return int(episode.maximum_generation) - pre


def _population_features(population, reproduction_threshold: float) -> np.ndarray:
    alive = np.asarray(population.alive, dtype=bool)
    count = max(int(np.sum(alive)), 1)
    return np.asarray(
        [
            np.mean(alive),
            np.sum(np.asarray(population.energy) * alive) / count / reproduction_threshold,
            float(population.population_change_ema),
            float(population.birth_rate_ema),
            float(population.death_rate_ema),
            np.sum(np.asarray(population.intake_ema) * alive) / count,
        ],
        dtype=np.float64,
    )


def _observable_stress(episodes, reproduction_threshold: float) -> tuple[float, float]:
    before = []
    after = []
    credit_tv = []
    for episode in episodes:
        if episode.shock_population is None or episode.final_population is None:
            raise RuntimeError("qualification requires captured populations")
        before.append(_population_features(episode.shock_population, reproduction_threshold))
        after.append(_population_features(episode.final_population, reproduction_threshold))
        pre_credit = np.asarray(episode.shock_population.operator_success_ema) * np.asarray(
            episode.shock_population.operator_evidence_ema
        )
        post_credit = np.asarray(episode.final_population.operator_success_ema) * np.asarray(
            episode.final_population.operator_evidence_ema
        )
        pre_credit = pre_credit / max(float(np.sum(pre_credit)), 1e-8)
        post_credit = post_credit / max(float(np.sum(post_credit)), 1e-8)
        credit_tv.append(0.5 * np.sum(np.abs(pre_credit - post_credit)))
    before = np.stack(before)
    after = np.stack(after)
    pooled = np.sqrt((np.var(before, axis=0, ddof=1) + np.var(after, axis=0, ddof=1)) / 2 + 1e-8)
    smd = np.max(np.abs(np.mean(after, axis=0) - np.mean(before, axis=0)) / pooled)
    return float(smd), float(np.mean(credit_tv))


def qualify_multiplier(founder_index_path: str | Path, multiplier: float) -> dict:
    index = load_founder_index(founder_index_path, verify_artifacts=False)
    founders = _founders(index)["training"]
    manifest = _manifest("training", founders, (4_101, 4_102), multiplier)
    config = SimulatorConfig()
    common = dict(
        capture_finalists=True,
        numerical_repeats=3,
        founder_index_path=founder_index_path,
    )
    clone = evaluate_manifest(manifest, config, fixed_r4_policy(R4_CLONE), **common)
    standard = evaluate_manifest(
        manifest, config, fixed_r4_policy(R4_STANDARD_PARAMETRIC), **common
    )
    exploration = evaluate_manifest(manifest, config, uniform_exploration_policy, **common)

    effects = _pair_effects(manifest, clone)
    effect_values = np.asarray([value for _, value in effects])
    harm_mean = float(np.mean(effect_values))
    harm_upper = _founder_bootstrap_upper(effects)
    clone_shock = _shock_episodes(manifest, clone)
    standard_shock = _shock_episodes(manifest, standard)
    exploration_shock = _shock_episodes(manifest, exploration)

    def viable(episodes):
        return {
            "survival": float(np.mean([bool(item.survived) for item in episodes])),
            "median_post_births": float(np.median([int(item.post_event_birth_count) for item in episodes])),
            "median_generation_gain": float(np.median([_generation_gain(item) for item in episodes])),
            "distinct_birth": int(sum(int(item.post_event_distinct_birth_count) for item in episodes)),
            "distinct_reproducer": int(
                sum(int(np.sum(np.asarray(item.post_event_resolved_distinct_success_count))) for item in episodes)
            ),
            "integrity": bool(all(bool(item.integrity_valid) for item in episodes)),
        }

    clone_viability = viable(clone_shock)
    standard_viability = viable(standard_shock)
    viable_policy = max((clone_viability, standard_viability), key=lambda item: (item["survival"], item["median_post_births"]))
    pre_resolved = np.sum(
        np.stack([np.asarray(item.pre_event_resolved_count) for item in exploration_shock]), axis=0
    )
    post_resolved = np.sum(
        np.stack([np.asarray(item.post_event_resolved_count) for item in exploration_shock]), axis=0
    )
    final_evidence = np.mean(
        np.stack([np.asarray(item.operator_evidence_ema) for item in exploration_shock]), axis=0
    )
    exercised = (pre_resolved > 0) & (post_resolved > 0) & (final_evidence >= 0.10)
    smd, credit_tv = _observable_stress(exploration_shock, config.reproduction_threshold)

    gates = {
        "harm": harm_mean <= -0.03 and harm_upper < 0.0,
        "viability": (
            viable_policy["survival"] >= 0.75
            and viable_policy["median_post_births"] >= 4
            and viable_policy["median_generation_gain"] >= 1
            and viable_policy["distinct_birth"] >= 1
            and viable_policy["distinct_reproducer"] >= 1
            and viable_policy["integrity"]
        ),
        "resolved_feedback": (
            int(np.sum(pre_resolved)) >= 24
            and int(np.sum(post_resolved)) >= 24
            and int(np.sum(exercised[1:])) >= 2
        ),
        "observable_stress": smd >= 0.5 or credit_tv >= 0.20,
        "integrity": bool(clone.integrity_valid & standard.integrity_valid & exploration.integrity_valid),
    }
    return {
        "multiplier": multiplier,
        "manifest_sha256": manifest_sha256(manifest),
        "clone_effect_mean": harm_mean,
        "clone_effect_founder_bootstrap_upper_95": harm_upper,
        "clone_viability": clone_viability,
        "standard_viability": standard_viability,
        "pre_resolved_by_operator": pre_resolved.tolist(),
        "post_resolved_by_operator": post_resolved.tolist(),
        "final_evidence_mean": final_evidence.tolist(),
        "observable_feature_max_smd": smd,
        "observable_credit_tv": credit_tv,
        "gates": gates,
        "passed": all(gates.values()),
    }


def run_qualification(founder_index_path: str | Path) -> dict:
    results = []
    selected = None
    for multiplier in MULTIPLIER_GRID:
        result = qualify_multiplier(founder_index_path, multiplier)
        results.append(result)
        if result["passed"]:
            selected = multiplier
            break
    index = load_founder_index(founder_index_path, verify_artifacts=False)
    return {
        "schema_version": 1,
        "status": "qualified" if selected is not None else "stop_no_qualified_multiplier",
        "selected_multiplier": selected,
        "multiplier_grid": list(MULTIPLIER_GRID),
        "founder_index_sha256": founder_index_sha256(index),
        "simulator_source_sha256": simulator_source_sha256(),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--founder-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace qualification evidence")
    result = run_qualification(args.founder_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected_multiplier": result["selected_multiplier"]}))


if __name__ == "__main__":
    main()
