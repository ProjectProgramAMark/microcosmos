"""Frozen manifests, catastrophe transforms, and scores for Evo²-Ecosystem."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp

from microcosmos.ecology import make_periodic_resource_patch
from microcosmos.heredity import update_population_change_ema
from microcosmos.rng import RNGTag, keys_for_identities
from microcosmos.structs.population import EcosystemState


PILOT_SCHEMA_VERSION = 1
SCHEMA_VERSION = 2
R4_SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = frozenset({PILOT_SCHEMA_VERSION, SCHEMA_VERSION, R4_SCHEMA_VERSION})
CONTROLLER_LAYOUT = "cppn-4x1-15n-30c-v1"
_PARTITIONS = frozenset({"calibration", "training", "development", "sealed_final"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EventKind(str, Enum):
    """The only event kinds admitted by the frozen protocol."""

    NULL = "null"
    RESOURCE_RELOCATION = "resource_relocation"
    RANDOM_BOTTLENECK = "random_bottleneck"
    DOMINANT_FOUNDER_LINEAGE_CULL = "dominant_founder_lineage_cull"
    ACTUATOR_INJURY = "actuator_injury"
    ACTUATION_COST_SHIFT = "actuation_cost_shift"


@dataclass(frozen=True)
class NullEventParameters:
    """The matched stable-world event has no parameters."""


@dataclass(frozen=True)
class ResourceRelocationParameters:
    """A periodic paraboloid resource patch installed at the event boundary."""

    center: tuple[float, float]
    radius: float
    peak_capacity: float
    peak_regeneration: float
    stock_fraction: float

    def __post_init__(self) -> None:
        if not isinstance(self.center, tuple) or len(self.center) != 2:
            raise ValueError("center must be a pair")
        _require_finite_pair("center", self.center)
        _require_positive_finite("radius", self.radius)
        _require_nonnegative_finite("peak_capacity", self.peak_capacity)
        _require_positive_finite("peak_regeneration", self.peak_regeneration)
        _require_fraction("stock_fraction", self.stock_fraction)


@dataclass(frozen=True)
class RandomBottleneckParameters:
    """Fitness-independent removal probability for each living organism."""

    removal_fraction: float

    def __post_init__(self) -> None:
        _require_fraction("removal_fraction", self.removal_fraction)


@dataclass(frozen=True)
class DominantLineageCullParameters:
    """Maximum fraction of the total living population that may be culled."""

    maximum_removal_fraction: float

    def __post_init__(self) -> None:
        _require_fraction("maximum_removal_fraction", self.maximum_removal_fraction)


@dataclass(frozen=True)
class ActuatorInjuryParameters:
    """Persistent efficacy gains for local bending hinges in every organism."""

    hinge_gains: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hinge_gains, tuple) or not self.hinge_gains:
            raise ValueError("hinge_gains must be a non-empty tuple")
        for gain in self.hinge_gains:
            _require_fraction("hinge_gains value", gain)
        if all(gain == 1.0 for gain in self.hinge_gains):
            raise ValueError("actuator injury must weaken at least one hinge")


@dataclass(frozen=True)
class ActuationCostShiftParameters:
    """Persistent multiplier applied only to metabolic actuation cost."""

    multiplier: float

    def __post_init__(self) -> None:
        _require_positive_finite("multiplier", self.multiplier)
        if self.multiplier <= 1.0:
            raise ValueError("actuation cost shift multiplier must exceed one")


EventParameters: TypeAlias = (
    NullEventParameters | ResourceRelocationParameters | RandomBottleneckParameters | DominantLineageCullParameters | ActuatorInjuryParameters | ActuationCostShiftParameters
)


@dataclass(frozen=True)
class WorldScenario:
    """One paired experimental world and its single boundary event."""

    world_seed: int
    scenario_id: str
    pair_id: str
    scenario_family: str
    event_kind: EventKind
    event_step: int
    event_parameters: EventParameters
    founder_id: str | None = None
    founder_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_int("world_seed", self.world_seed)
        _require_identifier("scenario_id", self.scenario_id)
        _require_identifier("pair_id", self.pair_id)
        _require_identifier("scenario_family", self.scenario_family)
        if not isinstance(self.event_kind, EventKind):
            raise ValueError("event_kind must be an EventKind")
        _require_nonnegative_int("event_step", self.event_step)
        expected = {
            EventKind.NULL: NullEventParameters,
            EventKind.RESOURCE_RELOCATION: ResourceRelocationParameters,
            EventKind.RANDOM_BOTTLENECK: RandomBottleneckParameters,
            EventKind.DOMINANT_FOUNDER_LINEAGE_CULL: DominantLineageCullParameters,
            EventKind.ACTUATOR_INJURY: ActuatorInjuryParameters,
            EventKind.ACTUATION_COST_SHIFT: ActuationCostShiftParameters,
        }[self.event_kind]
        if not isinstance(self.event_parameters, expected):
            raise ValueError(f"{self.event_kind.value} requires {expected.__name__}")
        if (self.founder_id is None) != (self.founder_sha256 is None):
            raise ValueError("founder_id and founder_sha256 must be provided together")
        if self.founder_id is not None:
            _require_identifier("founder_id", self.founder_id)
            if not _is_sha256(self.founder_sha256):
                raise ValueError("founder_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True)
class ScenarioManifest:
    """Canonical manifest whose bytes define an experiment partition."""

    schema_version: int
    partition: str
    controller_layout: str
    simulator_config_sha256: str
    horizon: int
    chunk_steps: int
    worlds: tuple[WorldScenario, ...]

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}")
        if self.partition not in _PARTITIONS:
            raise ValueError(f"partition must be one of {sorted(_PARTITIONS)}")
        if self.controller_layout != CONTROLLER_LAYOUT:
            raise ValueError(f"controller_layout must be {CONTROLLER_LAYOUT!r}")
        if not isinstance(self.simulator_config_sha256, str) or not _SHA256.fullmatch(self.simulator_config_sha256):
            raise ValueError("simulator_config_sha256 must be a lowercase SHA-256")
        _require_positive_int("horizon", self.horizon)
        _require_positive_int("chunk_steps", self.chunk_steps)
        if self.horizon % self.chunk_steps != 0:
            raise ValueError("horizon must be divisible by chunk_steps")
        if not isinstance(self.worlds, tuple) or not self.worlds:
            raise ValueError("worlds must be a non-empty tuple")
        if any(not isinstance(world, WorldScenario) for world in self.worlds):
            raise ValueError("worlds must contain only WorldScenario values")
        if self.schema_version == PILOT_SCHEMA_VERSION and any(world.event_kind is EventKind.ACTUATOR_INJURY for world in self.worlds):
            raise ValueError("actuator_injury requires manifest schema_version 2")
        if self.schema_version == PILOT_SCHEMA_VERSION and any(world.founder_id is not None for world in self.worlds):
            raise ValueError("founder metadata requires manifest schema_version 2")
        if self.schema_version < R4_SCHEMA_VERSION and any(world.event_kind is EventKind.ACTUATION_COST_SHIFT for world in self.worlds):
            raise ValueError("actuation_cost_shift requires manifest schema_version 3")
        if self.schema_version >= SCHEMA_VERSION and any(world.founder_id is None for world in self.worlds):
            raise ValueError("schema_version 2+ worlds require founder metadata")
        scenario_ids = [world.scenario_id for world in self.worlds]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario_id values must be unique")
        for world in self.worlds:
            if not 0 < world.event_step < self.horizon:
                raise ValueError("event_step must be strictly inside the horizon")
            if world.event_step % self.chunk_steps != 0:
                raise ValueError("event_step must fall on a chunk boundary")


def _require_nonnegative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a non-empty portable identifier")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require_finite_pair(name: str, values: tuple[float, float]) -> None:
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain two finite numbers")


def _require_positive_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_nonnegative_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_fraction(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")


def _require_exact_keys(value: dict[str, Any], keys: set[str], context: str) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ValueError(f"invalid {context} keys; missing={missing}, extra={extra}")


def _parse_event_parameters(event_kind: EventKind, value: object) -> EventParameters:
    if not isinstance(value, dict):
        raise ValueError("event_parameters must be an object")
    if event_kind is EventKind.NULL:
        _require_exact_keys(value, set(), "null event_parameters")
        return NullEventParameters()
    if event_kind is EventKind.RESOURCE_RELOCATION:
        keys = {
            "center",
            "radius",
            "peak_capacity",
            "peak_regeneration",
            "stock_fraction",
        }
        _require_exact_keys(value, keys, "resource relocation event_parameters")
        center = value["center"]
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError("resource relocation center must be a JSON pair")
        return ResourceRelocationParameters(
            center=(center[0], center[1]),
            radius=value["radius"],
            peak_capacity=value["peak_capacity"],
            peak_regeneration=value["peak_regeneration"],
            stock_fraction=value["stock_fraction"],
        )
    if event_kind is EventKind.RANDOM_BOTTLENECK:
        _require_exact_keys(value, {"removal_fraction"}, "bottleneck event_parameters")
        return RandomBottleneckParameters(removal_fraction=value["removal_fraction"])
    if event_kind is EventKind.DOMINANT_FOUNDER_LINEAGE_CULL:
        _require_exact_keys(
            value,
            {"maximum_removal_fraction"},
            "dominant-lineage event_parameters",
        )
        return DominantLineageCullParameters(maximum_removal_fraction=value["maximum_removal_fraction"])
    if event_kind is EventKind.ACTUATION_COST_SHIFT:
        _require_exact_keys(value, {"multiplier"}, "actuation cost event_parameters")
        return ActuationCostShiftParameters(multiplier=value["multiplier"])
    _require_exact_keys(value, {"hinge_gains"}, "actuator injury event_parameters")
    hinge_gains = value["hinge_gains"]
    if not isinstance(hinge_gains, list):
        raise ValueError("actuator injury hinge_gains must be a JSON array")
    return ActuatorInjuryParameters(hinge_gains=tuple(hinge_gains))


def manifest_from_dict(value: dict[str, Any]) -> ScenarioManifest:
    """Parse a manifest while rejecting missing, extra, or ill-typed fields."""
    if not isinstance(value, dict):
        raise ValueError("manifest must be an object")
    keys = {
        "schema_version",
        "partition",
        "controller_layout",
        "simulator_config_sha256",
        "horizon",
        "chunk_steps",
        "worlds",
    }
    _require_exact_keys(value, keys, "manifest")
    raw_worlds = value["worlds"]
    if not isinstance(raw_worlds, list):
        raise ValueError("worlds must be a JSON array")
    schema_version = value["schema_version"]
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}")
    worlds = []
    world_keys = {
        "world_seed",
        "scenario_id",
        "pair_id",
        "scenario_family",
        "event_kind",
        "event_step",
        "event_parameters",
    }
    if schema_version >= SCHEMA_VERSION:
        world_keys |= {"founder_id", "founder_sha256"}
    for raw_world in raw_worlds:
        if not isinstance(raw_world, dict):
            raise ValueError("each world must be an object")
        _require_exact_keys(raw_world, world_keys, "world")
        try:
            event_kind = EventKind(raw_world["event_kind"])
        except (TypeError, ValueError) as error:
            raise ValueError("unknown event_kind") from error
        worlds.append(
            WorldScenario(
                world_seed=raw_world["world_seed"],
                scenario_id=raw_world["scenario_id"],
                pair_id=raw_world["pair_id"],
                scenario_family=raw_world["scenario_family"],
                event_kind=event_kind,
                event_step=raw_world["event_step"],
                event_parameters=_parse_event_parameters(event_kind, raw_world["event_parameters"]),
                founder_id=raw_world.get("founder_id"),
                founder_sha256=raw_world.get("founder_sha256"),
            )
        )
    return ScenarioManifest(
        schema_version=value["schema_version"],
        partition=value["partition"],
        controller_layout=value["controller_layout"],
        simulator_config_sha256=value["simulator_config_sha256"],
        horizon=value["horizon"],
        chunk_steps=value["chunk_steps"],
        worlds=tuple(worlds),
    )


def manifest_from_json_bytes(value: bytes) -> ScenarioManifest:
    """Decode strict UTF-8 JSON and validate the resulting manifest."""
    if not isinstance(value, bytes):
        raise ValueError("manifest JSON must be bytes")
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest is not valid UTF-8 JSON") from error
    return manifest_from_dict(decoded)


def _event_parameters_dict(parameters: EventParameters) -> dict[str, Any]:
    if isinstance(parameters, NullEventParameters):
        return {}
    if isinstance(parameters, ResourceRelocationParameters):
        return {
            "center": [parameters.center[0], parameters.center[1]],
            "radius": parameters.radius,
            "peak_capacity": parameters.peak_capacity,
            "peak_regeneration": parameters.peak_regeneration,
            "stock_fraction": parameters.stock_fraction,
        }
    if isinstance(parameters, RandomBottleneckParameters):
        return {"removal_fraction": parameters.removal_fraction}
    if isinstance(parameters, DominantLineageCullParameters):
        return {"maximum_removal_fraction": parameters.maximum_removal_fraction}
    if isinstance(parameters, ActuationCostShiftParameters):
        return {"multiplier": parameters.multiplier}
    return {"hinge_gains": list(parameters.hinge_gains)}


def _manifest_dict(manifest: ScenarioManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "partition": manifest.partition,
        "controller_layout": manifest.controller_layout,
        "simulator_config_sha256": manifest.simulator_config_sha256,
        "horizon": manifest.horizon,
        "chunk_steps": manifest.chunk_steps,
        "worlds": [_world_dict(world, manifest.schema_version) for world in manifest.worlds],
    }


def _world_dict(world: WorldScenario, schema_version: int) -> dict[str, Any]:
    value = {
        "world_seed": world.world_seed,
        "scenario_id": world.scenario_id,
        "pair_id": world.pair_id,
        "scenario_family": world.scenario_family,
        "event_kind": world.event_kind.value,
        "event_step": world.event_step,
        "event_parameters": _event_parameters_dict(world.event_parameters),
    }
    if schema_version >= SCHEMA_VERSION:
        value["founder_id"] = world.founder_id
        value["founder_sha256"] = world.founder_sha256
    return value


def canonical_manifest_bytes(manifest: ScenarioManifest) -> bytes:
    """Return deterministic compact JSON bytes for a validated manifest."""
    if not isinstance(manifest, ScenarioManifest):
        raise ValueError("manifest must be a ScenarioManifest")
    return json.dumps(
        _manifest_dict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(manifest: ScenarioManifest) -> str:
    """Hash the exact canonical manifest bytes."""
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EventRecord:
    """Compact event telemetry retained by the host evaluator."""

    event_code: jax.Array
    alive_before: jax.Array
    alive_after: jax.Array
    catastrophe_death_count: jax.Array
    targeted_founder_lineage: jax.Array


_EVENT_CODE = {
    EventKind.NULL: 0,
    EventKind.RESOURCE_RELOCATION: 1,
    EventKind.RANDOM_BOTTLENECK: 2,
    EventKind.DOMINANT_FOUNDER_LINEAGE_CULL: 3,
    EventKind.ACTUATOR_INJURY: 4,
    EventKind.ACTUATION_COST_SHIFT: 5,
}


def _event_record(
    kind: EventKind,
    alive_before: jax.Array,
    alive_after: jax.Array,
    targeted_founder_lineage: jax.Array | int = -1,
) -> EventRecord:
    return EventRecord(
        event_code=jnp.asarray(_EVENT_CODE[kind], dtype=jnp.int32),
        alive_before=alive_before.astype(jnp.int32),
        alive_after=alive_after.astype(jnp.int32),
        catastrophe_death_count=(alive_before - alive_after).astype(jnp.int32),
        targeted_founder_lineage=jnp.asarray(targeted_founder_lineage, dtype=jnp.int32),
    )


def apply_null_event(state: EcosystemState) -> tuple[EcosystemState, EventRecord]:
    """Apply the exact identity transform used by matched stable worlds."""
    alive = jnp.sum(state.population.alive).astype(jnp.int32)
    return state, _event_record(EventKind.NULL, alive, alive)


def apply_resource_relocation(
    state: EcosystemState,
    center: tuple[float, float] | jax.Array,
    radius: float | jax.Array,
    peak_capacity: float | jax.Array,
    peak_regeneration: float | jax.Array,
    stock_fraction: float | jax.Array,
) -> tuple[EcosystemState, EventRecord]:
    """Replace resource maps and initialize stock from the new capacity map."""
    capacity, regeneration = make_periodic_resource_patch(
        state.fields.grid_shape,
        center,
        radius,
        peak_capacity,
        peak_regeneration,
    )
    capacity = capacity.astype(state.resource_capacity_map.dtype)
    regeneration = regeneration.astype(state.resource_regeneration_map.dtype)
    fields = replace(
        state.fields,
        energy=capacity * jnp.asarray(stock_fraction, dtype=capacity.dtype),
    )
    alive = jnp.sum(state.population.alive).astype(jnp.int32)
    return (
        replace(
            state,
            fields=fields,
            resource_capacity_map=capacity,
            resource_regeneration_map=regeneration,
        ),
        _event_record(EventKind.RESOURCE_RELOCATION, alive, alive),
    )


def apply_actuator_injury(
    state: EcosystemState,
    hinge_gains: tuple[float, ...] | jax.Array,
) -> tuple[EcosystemState, EventRecord]:
    """Install a persistent world-level actuator regime without changing life state."""
    gains = jnp.asarray(hinge_gains, dtype=state.actuator_gain.dtype)
    if gains.ndim != 1 or gains.shape != state.actuator_gain.shape:
        raise ValueError("hinge_gains must exactly match the world's local-hinge actuator shape")
    alive = jnp.sum(state.population.alive).astype(jnp.int32)
    return (
        replace(state, actuator_gain=gains),
        _event_record(EventKind.ACTUATOR_INJURY, alive, alive),
    )


def apply_actuation_cost_shift(
    state: EcosystemState,
    multiplier: float | jax.Array,
) -> tuple[EcosystemState, EventRecord]:
    """Install a persistent metabolic cost shock without changing physics."""
    multiplier = jnp.asarray(multiplier, dtype=state.actuation_cost_multiplier.dtype)
    alive = jnp.sum(state.population.alive).astype(jnp.int32)
    population = replace(
        state.population,
        shock_ancestor_id=jnp.where(
            state.population.alive,
            state.population.individual_id,
            -1,
        ),
    )
    return (
        replace(state, population=population, actuation_cost_multiplier=multiplier),
        _event_record(EventKind.ACTUATION_COST_SHIFT, alive, alive),
    )


def _identity_uniforms(
    event_key: jax.Array,
    event_step: int | jax.Array,
    identities: jax.Array,
) -> jax.Array:
    keys = keys_for_identities(
        event_key,
        RNGTag.FUTURE_EVENT,
        event_step,
        identities,
    )
    return jax.vmap(lambda key: jax.random.uniform(key, dtype=jnp.float32))(keys)


def _apply_catastrophe_deaths(
    state: EcosystemState,
    killed: jax.Array,
    node_slot: jax.Array,
    kind: EventKind,
    targeted_founder_lineage: jax.Array | int = -1,
) -> tuple[EcosystemState, EventRecord]:
    population = state.population
    alive_before = jnp.sum(population.alive).astype(jnp.int32)
    alive = population.alive & ~killed
    alive_after = jnp.sum(alive).astype(jnp.int32)
    population = replace(
        population,
        alive=alive,
        energy=jnp.where(killed, 0.0, population.energy),
        population_change_ema=update_population_change_ema(
            population.population_change_ema,
            alive_after - alive_before,
            population.alive.shape[0],
        ),
    )
    node_alive = alive[node_slot]
    nodes = replace(
        state.nodes,
        active=node_alive,
        velocity=jnp.where(node_alive[:, None], state.nodes.velocity, 0.0),
    )
    return (
        replace(state, population=population, nodes=nodes),
        _event_record(
            kind,
            alive_before,
            alive_after,
            targeted_founder_lineage,
        ),
    )


def apply_random_bottleneck(
    state: EcosystemState,
    event_key: jax.Array,
    event_step: int | jax.Array,
    removal_fraction: float | jax.Array,
    node_slot: jax.Array,
) -> tuple[EcosystemState, EventRecord]:
    """Remove each living identity independently of slot, fitness, and lineage."""
    draws = _identity_uniforms(
        event_key,
        event_step,
        state.population.individual_id,
    )
    killed = state.population.alive & (draws < jnp.asarray(removal_fraction, dtype=draws.dtype))
    return _apply_catastrophe_deaths(
        state,
        killed,
        node_slot,
        EventKind.RANDOM_BOTTLENECK,
    )


def apply_dominant_founder_lineage_cull(
    state: EcosystemState,
    event_key: jax.Array,
    event_step: int | jax.Array,
    maximum_removal_fraction: float | jax.Array,
    node_slot: jax.Array,
) -> tuple[EcosystemState, EventRecord]:
    """Cull the largest founder lineage, capped by total living occupancy."""
    population = state.population
    alive = population.alive
    founder = population.founder_lineage_id
    same_founder = founder[:, None] == founder[None, :]
    abundance = jnp.sum(same_founder & alive[None, :], axis=1)
    maximum_abundance = jnp.max(jnp.where(alive, abundance, 0))
    sentinel = jnp.iinfo(jnp.int32).max
    target = jnp.min(jnp.where(alive & (abundance == maximum_abundance), founder, sentinel))
    target = jnp.where(jnp.any(alive), target, -1).astype(jnp.int32)
    eligible = alive & (founder == target)

    alive_count = jnp.sum(alive).astype(jnp.int32)
    cap = jnp.floor(alive_count.astype(jnp.float32) * jnp.asarray(maximum_removal_fraction, dtype=jnp.float32)).astype(jnp.int32)
    removal_count = jnp.minimum(jnp.sum(eligible), cap).astype(jnp.int32)

    draws = _identity_uniforms(
        event_key,
        event_step,
        population.individual_id,
    )
    id_sentinel = jnp.iinfo(population.individual_id.dtype).max
    identity_order = jnp.argsort(
        jnp.where(eligible, population.individual_id, id_sentinel),
        stable=True,
    )
    score_in_identity_order = jnp.where(eligible[identity_order], draws[identity_order], jnp.inf)
    score_order = jnp.argsort(score_in_identity_order, stable=True)
    victim_order = identity_order[score_order]
    selected_by_rank = jnp.arange(alive.shape[0]) < removal_count
    killed = jnp.zeros_like(alive).at[victim_order].set(selected_by_rank)
    killed &= eligible
    return _apply_catastrophe_deaths(
        state,
        killed,
        node_slot,
        EventKind.DOMINANT_FOUNDER_LINEAGE_CULL,
        target,
    )


def normalized_chunk_productivity(
    cumulative_reward: jax.Array,
    chunk_steps: int | jax.Array,
    dt: float | jax.Array,
    post_event_regeneration_map: jax.Array,
    epsilon: float = 1e-8,
) -> jax.Array:
    """Return the frozen absolute productivity ratio, clipped to ``[0, 1]``."""
    reward = jnp.asarray(cumulative_reward)
    denominator = jnp.asarray(chunk_steps, dtype=reward.dtype) * jnp.asarray(dt, dtype=reward.dtype)
    gross_productivity = reward / denominator
    reference = jnp.maximum(
        jnp.sum(post_event_regeneration_map),
        jnp.asarray(epsilon, dtype=reward.dtype),
    )
    return jnp.clip(gross_productivity / reference, 0.0, 1.0)


def post_event_productivity_auc(normalized_productivity: jax.Array) -> jax.Array:
    """Primary episode score: mean normalized post-event productivity."""
    values = jnp.asarray(normalized_productivity)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError("normalized_productivity must be a non-empty vector")
    return jnp.mean(values)


def aggregate_candidate_score(episode_scores: jax.Array) -> jax.Array:
    """Frozen search aggregate: arithmetic mean over manifest worlds."""
    values = jnp.asarray(episode_scores)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError("episode_scores must be a non-empty vector")
    return jnp.mean(values)


def shock_null_productivity_deficit(shocked: jax.Array, paired_null: jax.Array) -> jax.Array:
    """Mean non-negative post-event deficit relative to the paired null."""
    shocked, paired_null = _paired_productivity_vectors(shocked, paired_null)
    return jnp.mean(jnp.maximum(0.0, paired_null - shocked))


def immediate_resistance(shocked: jax.Array, paired_null: jax.Array, checkpoints: int) -> jax.Array:
    """Mean shocked-minus-null productivity over the first checkpoints."""
    shocked, paired_null = _paired_productivity_vectors(shocked, paired_null)
    _validate_window("checkpoints", checkpoints, shocked.shape[0])
    return jnp.mean(shocked[:checkpoints] - paired_null[:checkpoints])


def final_window_productivity(values: jax.Array, checkpoints: int) -> jax.Array:
    """Mean normalized productivity over the final frozen window."""
    values = jnp.asarray(values)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError("values must be a non-empty vector")
    _validate_window("checkpoints", checkpoints, values.shape[0])
    return jnp.mean(values[-checkpoints:])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RecoveryResult:
    """Sustained-recovery checkpoint with explicit censoring semantics."""

    estimable: jax.Array
    recovered: jax.Array
    checkpoint: jax.Array


def sustained_recovery(
    shocked: jax.Array,
    paired_null: jax.Array,
    *,
    threshold: float,
    consecutive_checkpoints: int,
    smoothing_window: int,
    reference_floor: float,
) -> RecoveryResult:
    """Return the first sustained paired recovery, or ``-1`` when censored.

    A checkpoint denotes the first smoothed checkpoint in the qualifying run.
    A world is not estimable when mean paired-null productivity is below the
    frozen reference floor. A non-recovered but estimable world is censored.
    """
    shocked, paired_null = _paired_productivity_vectors(shocked, paired_null)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    if not math.isfinite(reference_floor) or reference_floor < 0.0:
        raise ValueError("reference_floor must be finite and non-negative")
    _validate_window("smoothing_window", smoothing_window, shocked.shape[0])
    smoothed_count = shocked.shape[0] - smoothing_window + 1
    _validate_window("consecutive_checkpoints", consecutive_checkpoints, smoothed_count)

    kernel = jnp.ones(smoothing_window, dtype=shocked.dtype) / smoothing_window
    shocked_smooth = jnp.convolve(shocked, kernel, mode="valid")
    null_smooth = jnp.convolve(paired_null, kernel, mode="valid")
    qualifies = (null_smooth >= reference_floor) & (shocked_smooth >= threshold * null_smooth)
    run_kernel = jnp.ones(consecutive_checkpoints, dtype=jnp.int32)
    qualifying_runs = jnp.convolve(qualifies.astype(jnp.int32), run_kernel, mode="valid") == consecutive_checkpoints
    recovered = jnp.any(qualifying_runs)
    first_run = jnp.argmax(qualifying_runs).astype(jnp.int32)
    first_checkpoint = first_run + smoothing_window - 1
    estimable = jnp.mean(paired_null) >= reference_floor
    recovered &= estimable
    checkpoint = jnp.where(recovered, first_checkpoint, -1).astype(jnp.int32)
    return RecoveryResult(
        estimable=estimable,
        recovered=recovered,
        checkpoint=checkpoint,
    )


def _paired_productivity_vectors(shocked: jax.Array, paired_null: jax.Array) -> tuple[jax.Array, jax.Array]:
    shocked = jnp.asarray(shocked)
    paired_null = jnp.asarray(paired_null)
    if shocked.ndim != 1 or paired_null.ndim != 1 or shocked.shape != paired_null.shape or shocked.shape[0] == 0:
        raise ValueError("paired productivity values must be equal non-empty vectors")
    return shocked, paired_null


def _validate_window(name: str, value: int, available: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= available:
        raise ValueError(f"{name} must be an integer within [1, {available}]")
