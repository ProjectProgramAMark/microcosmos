"""Frozen r5 result assembly and standardized common-garden analysis.

This module intentionally contains only r5 composition.  Simulator execution,
lineage extraction, ablation policies, and bootstrap primitives remain owned by
the existing r4 modules.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from microcosmos.cppn import CPPNGenome
from microcosmos.heredity import fixed_r4_policy
from microcosmos.rollout import run_ecosystem_chunk
from microcosmos.structs.population import EcosystemState
from microcosmos.utils import displacement, periodic_boundary

from ..episode import (
    EpisodeResult,
    PairedManifestEvaluation,
    SimulatorConfig,
    _step_keys,
    build_environment,
    simulator_config_sha256,
)
from ..protocol import (
    EventKind,
    ScenarioManifest,
    manifest_sha256,
    normalized_chunk_productivity,
)
from ..heredity_adaptation_v4.analysis import (
    LineagePair,
    dominant_action_ablation,
    extract_lineage_pairs,
    founder_bootstrap_interval,
    lineage_ledger,
    no_credit_ablation,
    paired_episode_effects,
)


COMMON_GARDEN_STEPS = 2_000
COMMON_GARDEN_RESET_SEED = 71_001
COMMON_GARDEN_POSES: tuple[tuple[str, tuple[float, float]], ...] = (
    ("inward", (-1.0, 0.0)),
    ("outward", (1.0, 0.0)),
    ("clockwise_tangent", (0.0, 1.0)),
    ("counterclockwise_tangent", (0.0, -1.0)),
)
PRIMARY_EFFECT_THRESHOLD = 0.02
SHAM_NONINFERIORITY_MARGIN = -0.02


def _finite_float(value: object, label: str) -> float:
    result = float(np.asarray(value))
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _normalized_direction(direction: Sequence[float]) -> jax.Array:
    value = jnp.asarray(direction, dtype=jnp.float32)
    if value.shape != (2,):
        raise ValueError("pose direction must contain exactly two values")
    norm = float(np.linalg.norm(np.asarray(value)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("pose direction must be finite and nonzero")
    return value / norm


def common_garden_mouth_position(config: SimulatorConfig) -> jax.Array:
    """Return the preregistered mouth location at 75% patch capacity."""
    center = jnp.asarray(config.resource_patch_center, dtype=jnp.float32)
    return center + jnp.asarray((0.5 * config.resource_patch_radius, 0.0), dtype=jnp.float32)


def rigid_common_garden_positions(
    state: EcosystemState,
    config: SimulatorConfig,
    head_to_tail_direction: Sequence[float],
) -> jax.Array:
    """Rigidly place the sole living body at the frozen mouth and orientation.

    Minimal-image offsets preserve the reset body exactly even if its original
    placement crossed a periodic boundary.  The function changes no resource,
    lifecycle, genome, or controller state.
    """
    if config.max_creatures != 1 or config.initial_population != 1:
        raise ValueError("common garden requires one preallocated living organism")
    positions = jnp.asarray(state.nodes.position)
    if positions.shape != (config.nodes_per_creature, 2):
        raise ValueError("common-garden state has the wrong node layout")

    mouth = positions[0]
    offsets = displacement(config.grid_shape, positions, mouth)
    current_axis = offsets[-1]
    current_angle = jnp.arctan2(current_axis[1], current_axis[0])
    target = _normalized_direction(head_to_tail_direction)
    target_angle = jnp.arctan2(target[1], target[0])
    angle = target_angle - current_angle
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    rotation = jnp.asarray(((cosine, -sine), (sine, cosine)), dtype=positions.dtype)
    rotated = offsets @ rotation.T
    return periodic_boundary(
        config.grid_shape,
        common_garden_mouth_position(config) + rotated,
    )


def prepare_common_garden_state(
    env,
    state: EcosystemState,
    config: SimulatorConfig,
    *,
    multiplier: float,
    head_to_tail_direction: Sequence[float],
) -> EcosystemState:
    """Apply only the frozen analysis-side rigid pose and post-shock cost."""
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("common-garden multiplier must be finite and positive")
    positions = rigid_common_garden_positions(state, config, head_to_tail_direction)
    nodes = replace(
        state.nodes,
        position=positions,
        velocity=jnp.zeros_like(state.nodes.velocity),
        debug_vector=jnp.zeros_like(state.nodes.debug_vector),
    )
    edges = replace(state.edges, theta=env._edge_theta(positions))
    return replace(
        state,
        nodes=nodes,
        edges=edges,
        actuation_cost_multiplier=jnp.asarray(multiplier, dtype=jnp.float32),
    )


def _common_garden_runner(config: SimulatorConfig):
    garden_config = replace(config, max_creatures=1, initial_population=1)
    env = build_environment(
        garden_config,
        fixed_r4_policy(0),
        horizon=COMMON_GARDEN_STEPS,
        heredity_contract="r4",
        credit_chunk_steps=500,
    )
    compiled = jax.jit(lambda current, keys: run_ecosystem_chunk(env, current, keys))
    return garden_config, env, compiled


def _common_garden_score(
    config: SimulatorConfig,
    env,
    compiled,
    genome: CPPNGenome,
    *,
    multiplier: float,
    direction: Sequence[float],
) -> float:
    root_key = jax.random.PRNGKey(COMMON_GARDEN_RESET_SEED)
    _, state = env.reset(root_key, founder_genome=genome)
    state = prepare_common_garden_state(
        env,
        state,
        config,
        multiplier=multiplier,
        head_to_tail_direction=direction,
    )
    keys = _step_keys(root_key, 0, COMMON_GARDEN_STEPS)
    _, metrics = compiled(state, keys)
    integrity_valid = (
        metrics.finite
        & metrics.identity_valid
        & metrics.events_valid
        & metrics.infrastructure_valid
        & (jnp.sum(metrics.operator_counts) == metrics.birth_count)
        & (metrics.policy_violation_count == 0)
    )
    if not bool(np.asarray(integrity_valid)):
        raise ValueError("common-garden rollout failed an integrity invariant")
    return _finite_float(
        normalized_chunk_productivity(
            metrics.cumulative_reward,
            COMMON_GARDEN_STEPS,
            env.dt,
            state.resource_regeneration_map,
        ),
        "common-garden productivity",
    )


def common_garden_analysis(
    pairs: tuple[LineagePair, ...],
    multiplier: float,
    *,
    config: SimulatorConfig | None = None,
) -> dict[str, object]:
    """Compare lineage-matched genomes across the four frozen poses.

    Pose effects are averaged within a lineage pair before founder-first
    aggregation; poses are therefore repeated conditions, not replicates.
    """
    if not pairs:
        raise ValueError("common-garden analysis requires lineage pairs")
    base_config = SimulatorConfig() if config is None else config
    garden_config, env, compiled = _common_garden_runner(base_config)
    observations: list[tuple[str, float]] = []
    records = []
    for pair in pairs:
        pose_records = []
        pose_deltas = []
        for pose_name, direction in COMMON_GARDEN_POSES:
            ancestor = _common_garden_score(
                garden_config,
                env,
                compiled,
                pair.ancestor,
                multiplier=multiplier,
                direction=direction,
            )
            descendant = _common_garden_score(
                garden_config,
                env,
                compiled,
                pair.descendant,
                multiplier=multiplier,
                direction=direction,
            )
            delta = descendant - ancestor
            pose_deltas.append(delta)
            pose_records.append(
                {
                    "pose": pose_name,
                    "head_to_tail_direction": list(direction),
                    "ancestor_score": ancestor,
                    "descendant_score": descendant,
                    "delta": delta,
                }
            )
        lineage_delta = float(np.mean(pose_deltas))
        observations.append((pair.founder_id, lineage_delta))
        records.append(
            {
                "founder_id": pair.founder_id,
                "ancestor_id": pair.ancestor_id,
                "descendant_id": pair.descendant_id,
                "pose_records": pose_records,
                "lineage_mean_delta": lineage_delta,
            }
        )
    return {
        "schema_version": 1,
        "reset_seed": COMMON_GARDEN_RESET_SEED,
        "steps": COMMON_GARDEN_STEPS,
        "multiplier": float(multiplier),
        "mouth_position": np.asarray(common_garden_mouth_position(garden_config)).tolist(),
        "mouth_capacity_fraction": 0.75,
        "poses_are_repeated_conditions": True,
        "records": records,
        "lineage_ledger": lineage_ledger(pairs),
        "founder_first_interval": founder_bootstrap_interval(observations),
    }


def _event_effects(
    manifest: ScenarioManifest,
    episodes: Sequence[EpisodeResult],
    *,
    event_kind: EventKind,
) -> list[tuple[str, float]]:
    return [
        (str(world.founder_id), _finite_float(episode.primary_score, "episode score"))
        for world, episode in zip(manifest.worlds, episodes, strict=True)
        if world.event_kind is event_kind
    ]


def _shock_minus_sham_effects(
    manifest: ScenarioManifest,
    episodes: Sequence[EpisodeResult],
) -> list[tuple[str, float]]:
    grouped: dict[str, dict[EventKind, tuple[str, float]]] = {}
    for world, episode in zip(manifest.worlds, episodes, strict=True):
        grouped.setdefault(world.pair_id, {})[world.event_kind] = (
            str(world.founder_id),
            _finite_float(episode.primary_score, "episode score"),
        )
    effects = []
    for pair_id in sorted(grouped):
        values = grouped[pair_id]
        null = values.get(EventKind.NULL)
        shocks = [value for kind, value in values.items() if kind is not EventKind.NULL]
        if null is None or len(shocks) != 1 or shocks[0][0] != null[0]:
            raise ValueError(f"manifest pair {pair_id!r} is not one matched sham/shock pair")
        effects.append((null[0], shocks[0][1] - null[1]))
    return effects


def primary_result_summary(
    manifest: ScenarioManifest,
    primary: PairedManifestEvaluation,
    clone: PairedManifestEvaluation,
) -> dict[str, object]:
    """Compute the preregistered four-part r5 primary decision."""
    if not bool(np.asarray(primary.integrity_valid)) or not bool(np.asarray(clone.integrity_valid)):
        raise ValueError("primary analysis requires integrity-valid evaluations")
    primary_effects = paired_episode_effects(
        manifest,
        primary.episodes,
        primary.ancestor_episodes,
        shocked_only=True,
    )
    sham_candidate = _event_effects(manifest, primary.episodes, event_kind=EventKind.NULL)
    sham_ancestor = _event_effects(manifest, primary.ancestor_episodes, event_kind=EventKind.NULL)
    if [item[0] for item in sham_candidate] != [item[0] for item in sham_ancestor]:
        raise ValueError("candidate and ancestor sham founders do not match")
    sham_effects = [
        (candidate[0], candidate[1] - ancestor[1])
        for candidate, ancestor in zip(sham_candidate, sham_ancestor, strict=True)
    ]
    clone_effects = _shock_minus_sham_effects(manifest, clone.episodes)
    primary_interval = founder_bootstrap_interval(primary_effects)
    sham_interval = founder_bootstrap_interval(sham_effects)
    clone_interval = founder_bootstrap_interval(clone_effects)
    gates = {
        "clone_shock_harm": clone_interval["upper_95"] < 0.0,
        "minimum_primary_mean": primary_interval["mean"] >= PRIMARY_EFFECT_THRESHOLD,
        "positive_primary_interval": primary_interval["lower_95"] > 0.0,
        "sham_noninferiority": sham_interval["lower_95"] > SHAM_NONINFERIORITY_MARGIN,
    }
    return {
        "primary_shocked_auc_delta": primary_interval,
        "sham_absolute_auc_delta": sham_interval,
        "clone_shock_minus_sham": clone_interval,
        "thresholds": {
            "primary_mean": PRIMARY_EFFECT_THRESHOLD,
            "sham_lower_bound": SHAM_NONINFERIORITY_MARGIN,
            "clone_upper_bound": 0.0,
            "primary_lower_bound": 0.0,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _ablation_summary(
    manifest: ScenarioManifest,
    primary: PairedManifestEvaluation,
    ablation: PairedManifestEvaluation,
) -> dict[str, object]:
    if not bool(np.asarray(primary.integrity_valid)) or not bool(
        np.asarray(ablation.integrity_valid)
    ):
        raise ValueError("ablation analysis requires integrity-valid evaluations")
    effects = paired_episode_effects(
        manifest,
        primary.episodes,
        ablation.episodes,
        shocked_only=True,
    )
    return founder_bootstrap_interval(effects)


def _write_json_once(path: Path, value: Mapping[str, object]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to replace final analysis at {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_final_analysis(
    manifest: ScenarioManifest,
    primary: PairedManifestEvaluation,
    clone: PairedManifestEvaluation,
    *,
    output_path: str | Path | None = None,
    lineage_pairs: tuple[LineagePair, ...] = (),
    multiplier: float | None = None,
    ablations: Mapping[str, PairedManifestEvaluation] | None = None,
    adaptive_eligible: bool = False,
) -> dict[str, object]:
    """Assemble the frozen primary and optional post-primary mechanism result.

    Optional evidence is evaluated only when the primary passes.  Supplying no
    optional inputs records an explicit skip rather than changing the primary.
    """
    primary_result = primary_result_summary(manifest, primary, clone)
    result: dict[str, object] = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256(manifest),
        "simulator_config_sha256": simulator_config_sha256(SimulatorConfig()),
        "primary": primary_result,
        "mechanism": {"status": "skipped_primary_not_positive"},
    }
    if primary_result["passed"]:
        mechanism: dict[str, object] = {"status": "skipped_not_requested"}
        if lineage_pairs:
            if not adaptive_eligible:
                raise ValueError(
                    "lineage mechanism analysis requires an adaptive-eligible finalist"
                )
            if multiplier is None:
                raise ValueError("lineage pairs require the frozen disturbance multiplier")
            required_ablations = {
                "dominant_action_ablation",
                "no_credit_ablation",
            }
            if set(ablations or {}) != required_ablations:
                raise ValueError(
                    "mechanism analysis requires exactly the two frozen ablations"
                )
            common_garden = common_garden_analysis(lineage_pairs, multiplier)
            common_garden_positive = (
                common_garden["founder_first_interval"]["lower_95"] > 0.0  # type: ignore[index]
            )
            mechanism = {
                "status": "completed",
                "adaptive_eligible": True,
                "common_garden": common_garden,
                "ablations": {},
            }
            primary_mean = primary_result["primary_shocked_auc_delta"]["mean"]  # type: ignore[index]
            ablation_supported = False
            for name, evaluation in sorted((ablations or {}).items()):
                summary = _ablation_summary(
                    manifest,
                    primary,
                    evaluation,
                )
                removes_half = summary["mean"] >= 0.5 * primary_mean
                positive_interval = summary["lower_95"] > 0.0
                mechanism["ablations"][name] = {  # type: ignore[index]
                    **summary,
                    "removes_at_least_half_primary_gain": removes_half,
                    "positive_finalist_minus_ablation_interval": positive_interval,
                }
                ablation_supported |= removes_half or positive_interval
            mechanism["claim_supported"] = (  # type: ignore[index]
                common_garden_positive and ablation_supported
            )
        result["mechanism"] = mechanism
    if output_path is not None:
        _write_json_once(Path(output_path), result)
    return result


__all__ = [
    "COMMON_GARDEN_POSES",
    "LineagePair",
    "common_garden_analysis",
    "common_garden_mouth_position",
    "dominant_action_ablation",
    "extract_lineage_pairs",
    "no_credit_ablation",
    "prepare_common_garden_state",
    "primary_result_summary",
    "rigid_common_garden_positions",
    "run_final_analysis",
]
