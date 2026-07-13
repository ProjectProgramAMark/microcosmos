"""Finalist-only r4 lineage, common-garden, ablation, and interval analyses."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from microcosmos.cppn import CPPNGenome
from microcosmos.heredity import R4OffspringPolicy, fixed_r4_policy
from microcosmos.rollout import run_ecosystem_chunk

from ..episode import EpisodeResult, ManifestEvaluation, SimulatorConfig, build_environment, _step_keys
from ..protocol import EventKind, ScenarioManifest, normalized_chunk_productivity


COMMON_GARDEN_STEPS = 2_000
COMMON_GARDEN_SEEDS = (71_001, 71_002, 71_003)
BOOTSTRAP_REPLICATES = 10_000


@dataclass(frozen=True)
class LineagePair:
    founder_id: str
    ancestor_id: int
    descendant_id: int
    ancestor: CPPNGenome
    descendant: CPPNGenome


def no_credit_ablation(policy: R4OffspringPolicy) -> R4OffspringPolicy:
    """Hold learned code fixed while neutralizing only online credit inputs."""
    def ablated(parent, parent_stats, population_stats, context):
        neutral = replace(
            population_stats,
            operator_success_ema=jnp.full(6, 0.5, dtype=jnp.float32),
            operator_usage_ema=jnp.zeros(6, dtype=jnp.float32),
            operator_evidence_ema=jnp.zeros(6, dtype=jnp.float32),
        )
        return policy(parent, parent_stats, neutral, context)

    return ablated


def dominant_action_ablation(operator: int) -> R4OffspringPolicy:
    return fixed_r4_policy(operator)


def _genome_at(population, slot: int) -> CPPNGenome:
    return CPPNGenome(
        population.genome.node_genes[slot],
        population.genome.connection_genes[slot],
    )


def _genome_sha256(genome: CPPNGenome) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(genome.node_genes).tobytes())
    digest.update(np.asarray(genome.connection_genes).tobytes())
    return digest.hexdigest()


def extract_lineage_pairs(
    manifest: ScenarioManifest,
    evaluation: ManifestEvaluation,
) -> tuple[LineagePair, ...]:
    """Choose one living post-shock descendant for each retained shock ancestor."""
    pairs = []
    for world, episode in zip(manifest.worlds, evaluation.episodes, strict=True):
        if world.event_kind is EventKind.NULL:
            continue
        before = episode.shock_population
        after = episode.final_population
        if before is None or after is None:
            raise ValueError("lineage extraction requires capture_finalists=True")
        before_ids = np.asarray(before.individual_id)
        after_alive = np.asarray(after.alive)
        after_ancestor = np.asarray(after.shock_ancestor_id)
        after_ids = np.asarray(after.individual_id)
        for ancestor_id in sorted(set(after_ancestor[after_alive]) - {-1}):
            ancestor_slots = np.flatnonzero(np.asarray(before.alive) & (before_ids == ancestor_id))
            descendant_slots = np.flatnonzero(
                after_alive
                & (after_ancestor == ancestor_id)
                & (after_ids >= int(before.next_individual_id))
            )
            if len(ancestor_slots) != 1 or not len(descendant_slots):
                continue
            descendant_slot = int(descendant_slots[np.argmax(after_ids[descendant_slots])])
            pairs.append(
                LineagePair(
                    founder_id=str(world.founder_id),
                    ancestor_id=int(ancestor_id),
                    descendant_id=int(after_ids[descendant_slot]),
                    ancestor=_genome_at(before, int(ancestor_slots[0])),
                    descendant=_genome_at(after, descendant_slot),
                )
            )
    return tuple(pairs)


def lineage_ledger(pairs: tuple[LineagePair, ...]) -> list[dict[str, object]]:
    return [
        {
            "founder_id": pair.founder_id,
            "ancestor_id": pair.ancestor_id,
            "descendant_id": pair.descendant_id,
            "ancestor_genome_sha256": _genome_sha256(pair.ancestor),
            "descendant_genome_sha256": _genome_sha256(pair.descendant),
            "genome_changed": _genome_sha256(pair.ancestor) != _genome_sha256(pair.descendant),
        }
        for pair in pairs
    ]


def _common_garden_runner():
    config = replace(
        SimulatorConfig(),
        max_creatures=1,
        initial_population=1,
    )
    env = build_environment(
        config,
        fixed_r4_policy(0),
        horizon=COMMON_GARDEN_STEPS,
        heredity_contract="r4",
        credit_chunk_steps=500,
    )
    compiled = jax.jit(
        lambda current, scan_keys: run_ecosystem_chunk(env, current, scan_keys)
    )
    return env, compiled


def _common_garden_score(env, compiled, genome: CPPNGenome, multiplier: float, seed: int) -> float:
    root_key = jax.random.PRNGKey(seed)
    _, state = env.reset(root_key, founder_genome=genome)
    state = replace(state, actuation_cost_multiplier=jnp.asarray(multiplier, dtype=jnp.float32))
    keys = _step_keys(root_key, 0, COMMON_GARDEN_STEPS)
    _, metrics = compiled(state, keys)
    return float(
        normalized_chunk_productivity(
            metrics.cumulative_reward,
            COMMON_GARDEN_STEPS,
            env.dt,
            state.resource_regeneration_map,
        )
    )


def founder_bootstrap_interval(observations, *, seed: int = 991_733) -> dict[str, float]:
    grouped = {}
    for founder_id, value in observations:
        grouped.setdefault(founder_id, []).append(value)
    founder_means = np.asarray([np.mean(values) for values in grouped.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(founder_means, size=(BOOTSTRAP_REPLICATES, len(founder_means)), replace=True)
    bootstrap = np.mean(samples, axis=1)
    return {
        "mean": float(np.mean(founder_means)),
        "lower_95": float(np.quantile(bootstrap, 0.025)),
        "upper_95": float(np.quantile(bootstrap, 0.975)),
    }


def common_garden_analysis(pairs: tuple[LineagePair, ...], multiplier: float) -> dict:
    observations = []
    records = []
    env, compiled = _common_garden_runner()
    for pair in pairs:
        ancestor_scores = [
            _common_garden_score(env, compiled, pair.ancestor, multiplier, seed)
            for seed in COMMON_GARDEN_SEEDS
        ]
        descendant_scores = [
            _common_garden_score(env, compiled, pair.descendant, multiplier, seed)
            for seed in COMMON_GARDEN_SEEDS
        ]
        delta = float(np.mean(descendant_scores) - np.mean(ancestor_scores))
        observations.append((pair.founder_id, delta))
        records.append(
            {
                "founder_id": pair.founder_id,
                "ancestor_id": pair.ancestor_id,
                "descendant_id": pair.descendant_id,
                "ancestor_scores": ancestor_scores,
                "descendant_scores": descendant_scores,
                "delta": delta,
            }
        )
    return {
        "records": records,
        "founder_first_interval": founder_bootstrap_interval(observations),
    }


def paired_episode_effects(
    manifest: ScenarioManifest,
    candidate: tuple[EpisodeResult, ...],
    reference: tuple[EpisodeResult, ...],
    *,
    shocked_only: bool,
) -> list[tuple[str, float]]:
    effects = []
    for world, left, right in zip(manifest.worlds, candidate, reference, strict=True):
        if shocked_only and world.event_kind is EventKind.NULL:
            continue
        effects.append((str(world.founder_id), float(left.primary_score - right.primary_score)))
    return effects
