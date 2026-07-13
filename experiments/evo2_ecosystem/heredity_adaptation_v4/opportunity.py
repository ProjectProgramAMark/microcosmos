"""Run the preregistered repeated-draw, founder-cross-fit r4 opportunity assay."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from microcosmos.cppn import CPPNGenome
from microcosmos.heredity import (
    R4_NUM_OPERATORS,
    fixed_r4_policy,
    r4_parent_statistics,
    r4_population_statistics,
)
from microcosmos.rollout import run_ecosystem_chunk

from ..episode import (
    SimulatorConfig,
    _apply_event,
    _resolve_manifest_founders,
    _step_keys,
    build_environment,
)
from ..founder_artifacts import load_founder_index
from ..protocol import EventKind, normalized_chunk_productivity
from .manifest_generator import _founders, _manifest


MUTATION_REPEATS = 8
ASSESSMENT_STEPS = 1_000
THRESHOLDS = tuple(index / 10 for index in range(1, 10))
BOOTSTRAP_REPLICATES = 10_000


def _eligible(env, state):
    pop = state.population
    lifecycle = env.lifecycle_config
    eligible = (
        pop.alive
        & (pop.age >= lifecycle.maturity_age)
        & (pop.energy >= lifecycle.reproduction_threshold)
        & (pop.energy >= lifecycle.reproduction_cost)
    )
    return jnp.any(eligible) & jnp.any(~pop.alive)


def _first_imminent_in_chunk(env, state, keys):
    def step(current, key):
        imminent = _eligible(env, current)
        _, updated, _, _, _ = env.step(key, current)
        return updated, imminent

    final, flags = jax.lax.scan(step, state, keys)
    return final, flags


def _find_imminent_state(env, compiled_probe, compiled_chunk, state, root_key, start, stop, chunk_steps):
    current = state
    first_imminent = None
    for chunk_start in range(start, stop, chunk_steps):
        keys = _step_keys(root_key, chunk_start, chunk_steps)
        final, flags = compiled_probe(current, keys)
        flags_host = np.asarray(flags)
        if first_imminent is None and np.any(flags_host):
            offset = int(np.argmax(flags_host))
            if offset == 0:
                first_imminent = current
            else:
                first_imminent, _ = compiled_chunk(current, keys[:offset])
        current = final
    return first_imminent, current


def _state_features(env, state) -> np.ndarray:
    pop = state.population
    lifecycle = env.lifecycle_config
    eligible = (
        pop.alive
        & (pop.age >= lifecycle.maturity_age)
        & (pop.energy >= lifecycle.reproduction_threshold)
        & (pop.energy >= lifecycle.reproduction_cost)
    )
    parent_slot = int(np.argmin(np.where(np.asarray(eligible), -np.asarray(pop.energy), np.inf)))
    genome = CPPNGenome(pop.genome.node_genes[parent_slot], pop.genome.connection_genes[parent_slot])
    parent = r4_parent_statistics(
        genome,
        pop.energy[parent_slot],
        pop.intake_ema[parent_slot],
        pop.age[parent_slot],
        lifecycle.reproduction_threshold,
        lifecycle.maximum_lifespan,
    )
    population = r4_population_statistics(
        pop.alive,
        pop.energy,
        pop.intake_ema,
        pop.population_change_ema,
        lifecycle.reproduction_threshold,
        pop.birth_rate_ema,
        pop.death_rate_ema,
        pop.operator_success_ema,
        pop.operator_usage_ema,
        pop.operator_evidence_ema,
    )
    return np.concatenate(
        [
            np.asarray([parent.node_fraction, parent.connection_fraction]),
            np.asarray([parent.energy_fraction, parent.intake_ema, parent.age_fraction]),
            np.asarray(
                [
                    population.alive_fraction,
                    population.mean_energy_fraction,
                    population.population_change_ema,
                    population.birth_rate_ema,
                    population.death_rate_ema,
                    population.mean_intake_ema,
                ]
            ),
            np.asarray(population.operator_success_ema),
            np.asarray(population.operator_usage_ema),
            np.asarray(population.operator_evidence_ema),
        ],
        dtype=np.float64,
    )


def _score_action(env, compiled_chunk, state, root_key, repeat_index):
    repeat_key = jax.random.fold_in(root_key, repeat_index + 91_000)
    first = int(state.time)
    keys = _step_keys(repeat_key, first, ASSESSMENT_STEPS)
    _, metrics = compiled_chunk(state, keys)
    score = normalized_chunk_productivity(
        metrics.cumulative_reward,
        ASSESSMENT_STEPS,
        env.dt,
        state.resource_regeneration_map,
    )
    resolved = metrics.resolved_success_count + metrics.resolved_failure_count
    return float(score), np.asarray(resolved, dtype=np.int64)


def collect_opportunity_records(founder_index_path: str | Path, multiplier: float):
    index = load_founder_index(founder_index_path, verify_artifacts=True)
    founders = _founders(index)["training"]
    manifest = _manifest("training", founders, (4_101, 4_102), multiplier)
    founder_genomes = _resolve_manifest_founders(manifest, founder_index_path)
    config = SimulatorConfig()
    base_env = build_environment(
        config,
        fixed_r4_policy(2),
        horizon=manifest.horizon,
        heredity_contract="r4",
        credit_chunk_steps=manifest.chunk_steps,
    )
    probe = jax.jit(lambda state, keys: _first_imminent_in_chunk(base_env, state, keys))
    base_chunk = jax.jit(lambda state, keys: run_ecosystem_chunk(base_env, state, keys))
    action_envs = [
        build_environment(
            config,
            fixed_r4_policy(action),
            horizon=manifest.horizon + ASSESSMENT_STEPS,
            heredity_contract="r4",
            credit_chunk_steps=manifest.chunk_steps,
        )
        for action in range(R4_NUM_OPERATORS)
    ]
    action_chunks = [
        jax.jit(lambda state, keys, env=env: run_ecosystem_chunk(env, state, keys))
        for env in action_envs
    ]
    records = []
    shock_worlds = [world for world in manifest.worlds if world.event_kind is EventKind.ACTUATION_COST_SHIFT]
    for world in shock_worlds:
        root_key = jax.random.PRNGKey(world.world_seed)
        _, reset_state = base_env.reset(root_key, founder_genome=founder_genomes[world.founder_id])
        pre_state, event_state = _find_imminent_state(
            base_env,
            probe,
            base_chunk,
            reset_state,
            root_key,
            0,
            world.event_step,
            manifest.chunk_steps,
        )
        event_state, _ = _apply_event(base_env, event_state, world, root_key)
        post_state, _ = _find_imminent_state(
            base_env,
            probe,
            base_chunk,
            event_state,
            root_key,
            world.event_step,
            manifest.horizon,
            manifest.chunk_steps,
        )
        if pre_state is None or post_state is None:
            raise RuntimeError(f"no imminent-birth state for {world.pair_id}")
        for context, state in (("pre", pre_state), ("post", post_state)):
            action_scores = np.zeros((R4_NUM_OPERATORS, MUTATION_REPEATS), dtype=np.float64)
            resolved = np.zeros((R4_NUM_OPERATORS, MUTATION_REPEATS), dtype=np.int64)
            for action, (env, chunk) in enumerate(zip(action_envs, action_chunks, strict=True)):
                for repeat_index in range(MUTATION_REPEATS):
                    score, outcome_counts = _score_action(
                        env, chunk, state, root_key, repeat_index
                    )
                    action_scores[action, repeat_index] = score
                    resolved[action, repeat_index] = outcome_counts[action]
            records.append(
                {
                    "founder_id": world.founder_id,
                    "pair_id": world.pair_id,
                    "context": context,
                    "features": _state_features(base_env, state).tolist(),
                    "action_scores": action_scores.tolist(),
                    "resolved": resolved.tolist(),
                }
            )
    return records


def crossfit_opportunity(records) -> dict:
    features = np.asarray([record["features"] for record in records], dtype=np.float64)
    rewards = np.asarray([record["action_scores"] for record in records], dtype=np.float64).mean(axis=2)
    founders = np.asarray([record["founder_id"] for record in records])
    contexts = np.asarray([record["context"] for record in records])
    held_effects = []
    selected_rules = []
    for held in sorted(set(founders)):
        train = founders != held
        test = ~train
        best_fixed = int(np.argmax(np.mean(rewards[train], axis=0)))
        best = None
        for feature in range(features.shape[1]):
            for threshold in THRESHOLDS:
                low = features[:, feature] < threshold
                for action_low in range(1, R4_NUM_OPERATORS):
                    for action_high in range(1, R4_NUM_OPERATORS):
                        chosen = np.where(low, action_low, action_high)
                        train_reward = np.mean(rewards[np.arange(len(records))[train], chosen[train]])
                        key = (train_reward, -feature, -threshold, -action_low, -action_high)
                        if best is None or key > best[0]:
                            best = (key, feature, threshold, action_low, action_high, best_fixed)
        _, feature, threshold, action_low, action_high, best_fixed = best
        chosen = np.where(features[test, feature] < threshold, action_low, action_high)
        selected = rewards[np.arange(len(records))[test], chosen]
        baseline = rewards[test, best_fixed]
        effects = selected - baseline
        held_effects.extend((held, float(value)) for value in effects)
        selected_rules.append(
            {
                "held_founder": held,
                "feature": feature,
                "threshold": threshold,
                "action_low": action_low,
                "action_high": action_high,
                "best_fixed_action": best_fixed,
            }
        )
    values = np.asarray([value for _, value in held_effects])
    grouped = defaultdict(list)
    for founder, value in held_effects:
        grouped[founder].append(value)
    founder_means = np.asarray([np.mean(value) for value in grouped.values()])
    rng = np.random.default_rng(882_341)
    samples = rng.choice(founder_means, size=(BOOTSTRAP_REPLICATES, len(founder_means)), replace=True)
    lower = float(np.quantile(np.mean(samples, axis=1), 0.025))
    exercised = {
        action
        for rule in selected_rules
        for action in (rule["action_low"], rule["action_high"])
    }
    resolved = np.asarray([record["resolved"] for record in records], dtype=np.int64)
    pre_resolved = np.sum(resolved[contexts == "pre"], axis=(0, 2))
    post_resolved = np.sum(resolved[contexts == "post"], axis=(0, 2))
    gates = {
        "mean_advantage": float(np.mean(values)) >= 0.03,
        "lower_interval": lower > 0.0,
        "two_nonclone_actions": len(exercised) >= 2,
        "outcome_evidence": all(
            pre_resolved[action] >= 8 and post_resolved[action] >= 8
            for action in exercised
        ),
    }
    return {
        "mean_advantage": float(np.mean(values)),
        "founder_bootstrap_lower_95": lower,
        "selected_rules": selected_rules,
        "exercised_nonclone_actions": sorted(exercised),
        "pre_resolved_by_operator": pre_resolved.tolist(),
        "post_resolved_by_operator": post_resolved.tolist(),
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--founder-index", type=Path, required=True)
    parser.add_argument("--multiplier", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace opportunity evidence")
    records = collect_opportunity_records(args.founder_index, args.multiplier)
    result = crossfit_opportunity(records)
    payload = {"schema_version": 1, "records": records, "summary": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "mean_advantage": result["mean_advantage"]}))


if __name__ == "__main__":
    main()
