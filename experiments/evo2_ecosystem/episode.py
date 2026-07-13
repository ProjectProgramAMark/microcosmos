"""Compact, provenance-tracked episode evaluation for Evo²-Ecosystem."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, cast

import jax
import jax.numpy as jnp
import numpy as np

from microcosmos.cppn import CPPNGenome
from microcosmos.gym import EcosystemEnv, LineTopology
from microcosmos.heredity import OffspringPolicy
from microcosmos.rng import RNGTag, derive_key
from microcosmos.rollout import (
    EcosystemChunkMetrics,
    combine_chunk_metrics,
    run_ecosystem_chunk,
)
from microcosmos.solver.config import PBD_SCHEME, PBD_SCHEME_NO_FLUID
from microcosmos.structs.population import EcosystemState, PopulationState

from .founder_artifacts import (
    FounderIndex,
    load_founder_artifact,
    load_founder_index,
)
from .protocol import (
    ActuatorInjuryParameters,
    DominantLineageCullParameters,
    EventKind,
    EventRecord,
    RandomBottleneckParameters,
    ResourceRelocationParameters,
    PILOT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ScenarioManifest,
    WorldScenario,
    aggregate_candidate_score,
    apply_actuator_injury,
    apply_dominant_founder_lineage_cull,
    apply_null_event,
    apply_random_bottleneck,
    apply_resource_relocation,
    normalized_chunk_productivity,
    post_event_productivity_auc,
)
from .frozen_config import CALIBRATED_ECOLOGY, NUMERICAL_REPEATS


_MANIFEST_TO_FOUNDER_PARTITION = {
    "calibration": "training",
    "training": "training",
    "development": "development",
    "sealed_final": "sealed",
}


@dataclass(frozen=True)
class SimulatorConfig:
    """Host-static simulator settings excluded from the evolvable policy."""

    max_creatures: int = 32
    initial_population: int = CALIBRATED_ECOLOGY["initial_population"]
    nodes_per_creature: int = 8
    node_spacing: float = 2.0
    bending_stiffness: float = 0.5
    grid_shape: tuple[int, int] = (64, 64)
    dt: float = 0.01
    fluid_enabled: bool = True
    resource_capacity: float = 1.0
    initial_resource: float = CALIBRATED_ECOLOGY["initial_resource_fraction"]
    resource_regeneration_rate: float = CALIBRATED_ECOLOGY["resource_regeneration_rate"]
    resource_diffusion_rate: float = 0.0
    resource_patch_center: tuple[float, float] = (48.0, 32.0)
    resource_patch_radius: float = 12.0
    initial_energy: float = 2.0
    birth_transfer_efficiency: float = CALIBRATED_ECOLOGY["birth_transfer_efficiency"]
    reproduction_threshold: float = CALIBRATED_ECOLOGY["reproduction_threshold"]
    reproduction_cost: float = CALIBRATED_ECOLOGY["reproduction_cost"]
    maturity_age: int = CALIBRATED_ECOLOGY["maturity_age"]
    maximum_lifespan: int = CALIBRATED_ECOLOGY["maximum_lifespan"]
    max_bending_delta: float = 0.35
    uptake_rate: float = CALIBRATED_ECOLOGY["uptake_rate"]
    assimilation_efficiency: float = CALIBRATED_ECOLOGY["assimilation_efficiency"]
    basal_metabolism: float = CALIBRATED_ECOLOGY["basal_metabolism"]
    actuation_power_coefficient: float = CALIBRATED_ECOLOGY["actuation_power_coefficient"]
    spawn_separation: float | None = None
    placement_candidates: int = 16
    position_margin: float = 1.0


def canonical_simulator_config_bytes(config: SimulatorConfig) -> bytes:
    """Return the canonical bytes named by a scenario manifest."""
    if not isinstance(config, SimulatorConfig):
        raise ValueError("config must be a SimulatorConfig")
    return json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def simulator_config_sha256(config: SimulatorConfig) -> str:
    return hashlib.sha256(canonical_simulator_config_bytes(config)).hexdigest()


def simulator_source_sha256() -> str:
    """Hash the Python source that defines the trusted simulator/evaluator."""
    root = Path(__file__).resolve().parents[2]
    paths = sorted(
        (*((root / "src" / "microcosmos").rglob("*.py")), *((root / "experiments" / "evo2_ecosystem").rglob("*.py"))),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_environment(
    config: SimulatorConfig,
    offspring_policy: OffspringPolicy,
    *,
    horizon: int,
) -> EcosystemEnv:
    """Build the fixed simulator while injecting only the heredity policy."""
    solver = PBD_SCHEME if config.fluid_enabled else PBD_SCHEME_NO_FLUID
    return EcosystemEnv(
        topology=LineTopology(
            num_nodes=config.nodes_per_creature,
            spacing=config.node_spacing,
            bending_stiffness=config.bending_stiffness,
        ),
        max_creatures=config.max_creatures,
        initial_population=config.initial_population,
        grid_shape=config.grid_shape,
        dt=config.dt,
        max_steps=horizon,
        solver_config=solver,
        resource_capacity=config.resource_capacity,
        initial_resource=config.initial_resource,
        resource_regeneration_rate=config.resource_regeneration_rate,
        resource_diffusion_rate=config.resource_diffusion_rate,
        resource_patch_center=config.resource_patch_center,
        resource_patch_radius=config.resource_patch_radius,
        initial_energy=config.initial_energy,
        birth_transfer_efficiency=config.birth_transfer_efficiency,
        reproduction_threshold=config.reproduction_threshold,
        reproduction_cost=config.reproduction_cost,
        maturity_age=config.maturity_age,
        maximum_lifespan=config.maximum_lifespan,
        max_bending_delta=config.max_bending_delta,
        uptake_rate=config.uptake_rate,
        assimilation_efficiency=config.assimilation_efficiency,
        basal_metabolism=config.basal_metabolism,
        actuation_power_coefficient=config.actuation_power_coefficient,
        spawn_separation=config.spawn_separation,
        placement_candidates=config.placement_candidates,
        position_margin=config.position_margin,
        offspring_policy=offspring_policy,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodeResult:
    """Fixed-size search result; shock population is finalist-only."""

    post_event_productivity: jax.Array
    primary_score: jax.Array
    survived: jax.Array
    final_alive: jax.Array
    minimum_population: jax.Array
    maximum_generation: jax.Array
    birth_count: jax.Array
    natural_death_count: jax.Array
    operator_counts: jax.Array
    finite: jax.Array
    identity_valid: jax.Array
    events_valid: jax.Array
    infrastructure_valid: jax.Array
    operator_accounting_valid: jax.Array
    policy_violation_count: jax.Array
    integrity_valid: jax.Array
    event_record: EventRecord
    shock_population: PopulationState | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ManifestEvaluation:
    """One coherent median run plus its numerical-repeat audit."""

    episodes: tuple[EpisodeResult, ...]
    candidate_score: jax.Array
    integrity_valid: jax.Array
    repeat_scores: jax.Array
    selected_repeat_index: jax.Array


CompiledChunk = Callable[
    [EcosystemState, jax.Array],
    tuple[EcosystemState, EcosystemChunkMetrics],
]


def _step_keys(
    root_key: jax.Array,
    first_step: int,
    chunk_steps: int,
) -> jax.Array:
    steps = jnp.arange(first_step, first_step + chunk_steps, dtype=jnp.uint32)
    return jax.vmap(lambda step: derive_key(root_key, RNGTag.ENVIRONMENT, step))(steps)


def _run_chunks(
    compiled_chunk: CompiledChunk,
    state: EcosystemState,
    root_key: jax.Array,
    *,
    first_step: int,
    stop_step: int,
    chunk_steps: int,
    collect_rewards: bool = False,
) -> tuple[EcosystemState, EcosystemChunkMetrics, tuple[jax.Array, ...]]:
    combined = None
    rewards = []
    for chunk_start in range(first_step, stop_step, chunk_steps):
        keys = _step_keys(root_key, chunk_start, chunk_steps)
        state, metrics = compiled_chunk(state, keys)
        combined = metrics if combined is None else combine_chunk_metrics(combined, metrics)
        if collect_rewards:
            rewards.append(metrics.cumulative_reward)
    if combined is None:
        raise ValueError("episode phases must each contain at least one chunk")
    return state, combined, tuple(rewards)


def _apply_event(
    env: EcosystemEnv,
    state: EcosystemState,
    world: WorldScenario,
    root_key: jax.Array,
) -> tuple[EcosystemState, EventRecord]:
    """Dispatch on the host, then compile exactly the selected event transform."""
    if world.event_kind is EventKind.NULL:
        return jax.jit(apply_null_event)(state)

    parameters = world.event_parameters
    if world.event_kind is EventKind.RESOURCE_RELOCATION:
        relocation = cast(ResourceRelocationParameters, parameters)
        apply = jax.jit(
            lambda current: apply_resource_relocation(
                current,
                relocation.center,
                relocation.radius,
                relocation.peak_capacity,
                relocation.peak_regeneration,
                relocation.stock_fraction,
            )
        )
        return apply(state)
    if world.event_kind is EventKind.RANDOM_BOTTLENECK:
        bottleneck = cast(RandomBottleneckParameters, parameters)
        apply = jax.jit(
            lambda current, key: apply_random_bottleneck(
                current,
                key,
                world.event_step,
                bottleneck.removal_fraction,
                env.node_slot,
            )
        )
        return apply(state, root_key)
    if world.event_kind is EventKind.ACTUATOR_INJURY:
        injury = cast(ActuatorInjuryParameters, parameters)
        apply = jax.jit(
            lambda current: apply_actuator_injury(
                current,
                injury.hinge_gains,
            )
        )
        return apply(state)
    lineage_cull = cast(DominantLineageCullParameters, parameters)
    apply = jax.jit(
        lambda current, key: apply_dominant_founder_lineage_cull(
            current,
            key,
            world.event_step,
            lineage_cull.maximum_removal_fraction,
            env.node_slot,
        )
    )
    return apply(state, root_key)


def _prepare_pre_event(
    env: EcosystemEnv,
    compiled_chunk: CompiledChunk,
    world: WorldScenario,
    *,
    chunk_steps: int,
    founder_genome: CPPNGenome | None = None,
) -> tuple[jax.Array, EcosystemState, EcosystemChunkMetrics]:
    root_key = jax.random.PRNGKey(world.world_seed)
    if founder_genome is None:
        # Preserve the schema-v1 pilot reset path exactly.
        _, state = env.reset(root_key)
    else:
        _, state = env.reset(root_key, founder_genome=founder_genome)
    state, pre_metrics, _ = _run_chunks(
        compiled_chunk,
        state,
        root_key,
        first_step=0,
        stop_step=world.event_step,
        chunk_steps=chunk_steps,
    )
    return root_key, state, pre_metrics


def _finish_world(
    env: EcosystemEnv,
    compiled_chunk: CompiledChunk,
    world: WorldScenario,
    root_key: jax.Array,
    pre_event_state: EcosystemState,
    pre_metrics: EcosystemChunkMetrics,
    *,
    horizon: int,
    chunk_steps: int,
    capture_finalist: bool,
) -> EpisodeResult:
    state = pre_event_state
    shock_population = state.population if capture_finalist else None
    state, event_record = _apply_event(env, state, world, root_key)
    post_event_regeneration_map = state.resource_regeneration_map
    state, post_metrics, post_rewards = _run_chunks(
        compiled_chunk,
        state,
        root_key,
        first_step=world.event_step,
        stop_step=horizon,
        chunk_steps=chunk_steps,
        collect_rewards=True,
    )
    metrics = combine_chunk_metrics(pre_metrics, post_metrics)
    productivity = jnp.stack(
        [
            normalized_chunk_productivity(
                reward,
                chunk_steps,
                env.dt,
                post_event_regeneration_map,
            )
            for reward in post_rewards
        ]
    )
    final_alive = jnp.sum(state.population.alive).astype(jnp.int32)
    operator_accounting_valid = jnp.sum(metrics.operator_counts) == metrics.birth_count
    integrity_valid = (
        metrics.finite
        & metrics.identity_valid
        & metrics.events_valid
        & metrics.infrastructure_valid
        & operator_accounting_valid
        & (metrics.policy_violation_count == 0)
    )
    return EpisodeResult(
        post_event_productivity=productivity,
        primary_score=post_event_productivity_auc(productivity),
        survived=final_alive > 0,
        final_alive=final_alive,
        minimum_population=metrics.minimum_alive,
        maximum_generation=metrics.maximum_generation,
        birth_count=metrics.birth_count,
        natural_death_count=metrics.death_count,
        operator_counts=metrics.operator_counts,
        finite=metrics.finite,
        identity_valid=metrics.identity_valid,
        events_valid=metrics.events_valid,
        infrastructure_valid=metrics.infrastructure_valid,
        operator_accounting_valid=operator_accounting_valid,
        policy_violation_count=metrics.policy_violation_count,
        integrity_valid=integrity_valid,
        event_record=event_record,
        shock_population=shock_population,
    )


def _run_world(
    env: EcosystemEnv,
    compiled_chunk: CompiledChunk,
    world: WorldScenario,
    *,
    horizon: int,
    chunk_steps: int,
    capture_finalist: bool,
    founder_genome: CPPNGenome | None = None,
) -> EpisodeResult:
    root_key, state, pre_metrics = _prepare_pre_event(
        env,
        compiled_chunk,
        world,
        chunk_steps=chunk_steps,
        founder_genome=founder_genome,
    )
    return _finish_world(
        env,
        compiled_chunk,
        world,
        root_key,
        state,
        pre_metrics,
        horizon=horizon,
        chunk_steps=chunk_steps,
        capture_finalist=capture_finalist,
    )


def run_world_scenario(
    config: SimulatorConfig,
    offspring_policy: OffspringPolicy,
    world: WorldScenario,
    *,
    horizon: int,
    chunk_steps: int,
    capture_finalist: bool = False,
    founder_index_path: str | Path | None = None,
    founder_partition: str | None = None,
) -> EpisodeResult:
    """Evaluate one world, resolving declared founders through a trusted index.

    The two founder arguments are required together only when ``world`` carries
    schema-v2 founder metadata. Worlds without founder metadata retain the
    schema-v1 varied-founder reset behavior.
    """
    if not isinstance(world, WorldScenario):
        raise ValueError("world must be a WorldScenario")
    if not isinstance(capture_finalist, bool):
        raise ValueError("capture_finalist must be a bool")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(chunk_steps, int) or isinstance(chunk_steps, bool) or chunk_steps <= 0:
        raise ValueError("chunk_steps must be a positive integer")
    if not 0 < world.event_step < horizon:
        raise ValueError("event_step must be strictly inside the horizon")
    if horizon % chunk_steps or world.event_step % chunk_steps:
        raise ValueError("horizon and event_step must fall on chunk boundaries")
    founder_genome = _resolve_world_founder(
        world,
        founder_index_path=founder_index_path,
        founder_partition=founder_partition,
    )
    env = build_environment(
        config,
        offspring_policy,
        horizon=horizon,
    )
    compiled_chunk = jax.jit(lambda state, keys: run_ecosystem_chunk(env, state, keys))
    return _run_world(
        env,
        compiled_chunk,
        world,
        horizon=horizon,
        chunk_steps=chunk_steps,
        capture_finalist=capture_finalist,
        founder_genome=founder_genome,
    )


def evaluate_manifest(
    manifest: ScenarioManifest,
    config: SimulatorConfig,
    offspring_policy: OffspringPolicy,
    *,
    capture_finalists: bool = False,
    numerical_repeats: int = NUMERICAL_REPEATS,
    founder_index_path: str | Path | None = None,
) -> ManifestEvaluation:
    """Select the coherent median of repeated full-manifest measurements."""
    _validate_manifest_inputs(
        manifest,
        config,
        capture_finalists,
        founder_index_path=founder_index_path,
    )
    if not isinstance(numerical_repeats, int) or isinstance(numerical_repeats, bool) or numerical_repeats <= 0 or numerical_repeats % 2 == 0:
        raise ValueError("numerical_repeats must be a positive odd integer")
    founders = _resolve_manifest_founders(manifest, founder_index_path)
    env = build_environment(config, offspring_policy, horizon=manifest.horizon)
    compiled_chunk = jax.jit(lambda state, keys: run_ecosystem_chunk(env, state, keys))
    evaluations = tuple(
        _evaluate_manifest_once(
            manifest,
            env,
            compiled_chunk,
            founders,
            capture_finalists=capture_finalists,
        )
        for _ in range(numerical_repeats)
    )
    scores = np.asarray([float(item.candidate_score) for item in evaluations])
    selected_index = int(np.argsort(scores, kind="stable")[numerical_repeats // 2])
    selected = evaluations[selected_index]
    return replace(
        selected,
        integrity_valid=jnp.all(jnp.stack([item.integrity_valid for item in evaluations])),
        repeat_scores=jnp.asarray(scores, dtype=jnp.float32),
        selected_repeat_index=jnp.asarray(selected_index, dtype=jnp.int32),
    )


def _validate_manifest_inputs(
    manifest: ScenarioManifest,
    config: SimulatorConfig,
    capture_finalists: bool,
    *,
    founder_index_path: str | Path | None,
) -> None:
    if not isinstance(manifest, ScenarioManifest):
        raise ValueError("manifest must be a ScenarioManifest")
    if manifest.simulator_config_sha256 != simulator_config_sha256(config):
        raise ValueError("manifest simulator config hash does not match config")
    if not isinstance(capture_finalists, bool):
        raise ValueError("capture_finalists must be a bool")
    if manifest.schema_version == SCHEMA_VERSION and founder_index_path is None:
        raise ValueError("schema_version 2 requires a trusted founder_index_path")
    if (
        manifest.schema_version == PILOT_SCHEMA_VERSION
        and founder_index_path is not None
    ):
        raise ValueError("schema_version 1 must not declare a founder_index_path")

    grouped: dict[str, list[WorldScenario]] = {}
    for world in manifest.worlds:
        grouped.setdefault(world.pair_id, []).append(world)
    for worlds in grouped.values():
        reference = worlds[0]
        if any(
            world.world_seed != reference.world_seed
            or world.event_step != reference.event_step
            for world in worlds
        ):
            raise ValueError("paired worlds must share world_seed and event_step")
        if any(
            (world.founder_id, world.founder_sha256)
            != (reference.founder_id, reference.founder_sha256)
            for world in worlds
        ):
            raise ValueError("paired worlds must share founder identity")


def _founder_partition(partition: str) -> str:
    try:
        return _MANIFEST_TO_FOUNDER_PARTITION[partition]
    except KeyError as error:
        raise ValueError(
            f"schema-version-2 partition {partition!r} has no founder-bank partition"
        ) from error


def _load_founders(
    worlds: tuple[WorldScenario, ...],
    *,
    founder_index_path: str | Path,
    expected_partition: str,
) -> dict[str, CPPNGenome]:
    """Load one trusted index and each unique declared founder exactly once."""
    path = Path(founder_index_path)
    index: FounderIndex = load_founder_index(path, verify_artifacts=False)
    founders: dict[str, CPPNGenome] = {}
    for world in worlds:
        if world.founder_id is None or world.founder_sha256 is None:
            raise ValueError("founder metadata is required when resolving founders")
        record = index.record(
            world.founder_id,
            expected_partition=expected_partition,
        )
        if record.artifact_sha256 != world.founder_sha256:
            raise ValueError(
                f"manifest founder hash mismatch for {world.founder_id!r}"
            )
        if world.founder_id not in founders:
            founders[world.founder_id] = load_founder_artifact(
                path.parent,
                record,
                expected_partition=expected_partition,
            )
    return founders


def _resolve_manifest_founders(
    manifest: ScenarioManifest,
    founder_index_path: str | Path | None,
) -> dict[str, CPPNGenome]:
    if manifest.schema_version == PILOT_SCHEMA_VERSION:
        return {}
    if founder_index_path is None:  # Defensive; validated before this boundary.
        raise ValueError("schema_version 2 requires a trusted founder_index_path")
    return _load_founders(
        manifest.worlds,
        founder_index_path=founder_index_path,
        expected_partition=_founder_partition(manifest.partition),
    )


def _resolve_world_founder(
    world: WorldScenario,
    *,
    founder_index_path: str | Path | None,
    founder_partition: str | None,
) -> CPPNGenome | None:
    has_founder = world.founder_id is not None
    if not has_founder:
        if founder_index_path is not None or founder_partition is not None:
            raise ValueError("worlds without founder metadata do not use a founder index")
        return None
    if founder_index_path is None or founder_partition is None:
        raise ValueError(
            "world founder metadata requires founder_index_path and founder_partition"
        )
    expected_partition = (
        _MANIFEST_TO_FOUNDER_PARTITION.get(founder_partition, founder_partition)
    )
    founders = _load_founders(
        (world,),
        founder_index_path=founder_index_path,
        expected_partition=expected_partition,
    )
    return founders[cast(str, world.founder_id)]


def _evaluate_manifest_once(
    manifest: ScenarioManifest,
    env: EcosystemEnv,
    compiled_chunk: CompiledChunk,
    founders: Mapping[str, CPPNGenome],
    *,
    capture_finalists: bool,
) -> ManifestEvaluation:
    grouped: dict[str, list[int]] = {}
    for index, world in enumerate(manifest.worlds):
        grouped.setdefault(world.pair_id, []).append(index)
    episodes: list[EpisodeResult | None] = [None] * len(manifest.worlds)
    for indices in grouped.values():
        reference = manifest.worlds[indices[0]]
        founder_genome = (
            None
            if reference.founder_id is None
            else founders[reference.founder_id]
        )
        root_key, state, pre_metrics = _prepare_pre_event(
            env,
            compiled_chunk,
            reference,
            chunk_steps=manifest.chunk_steps,
            founder_genome=founder_genome,
        )
        for index in indices:
            episodes[index] = _finish_world(
                env,
                compiled_chunk,
                manifest.worlds[index],
                root_key,
                state,
                pre_metrics,
                horizon=manifest.horizon,
                chunk_steps=manifest.chunk_steps,
                capture_finalist=capture_finalists,
            )
    completed = tuple(episode for episode in episodes if episode is not None)
    if len(completed) != len(manifest.worlds):
        raise RuntimeError("not every manifest world was evaluated")
    scores = jnp.stack([episode.primary_score for episode in completed])
    candidate_score = aggregate_candidate_score(scores)
    return ManifestEvaluation(
        episodes=completed,
        candidate_score=candidate_score,
        integrity_valid=jnp.all(jnp.stack([episode.integrity_valid for episode in completed])),
        repeat_scores=jnp.reshape(candidate_score, (1,)),
        selected_repeat_index=jnp.asarray(0, dtype=jnp.int32),
    )
