"""Memory-bounded ecosystem rollout chunks and compact metrics."""

from dataclasses import dataclass, replace
import functools

import jax
import jax.numpy as jnp
from tensorneat.common import I_INF

from microcosmos.cppn import (
    MAX_CONNECTIONS,
    MAX_NODES,
    population_numeric_valid,
)
from microcosmos.heredity import NUM_OPERATORS
from microcosmos.structs.population import EcosystemState


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=[
        "steps",
        "cumulative_reward",
        "birth_count",
        "death_count",
        "minimum_alive",
        "maximum_alive",
        "maximum_generation",
        "steps_at_capacity",
        "maximum_individual_id",
        "minimum_resource",
        "maximum_resource_excess",
        "minimum_live_energy",
        "finite",
        "identity_valid",
        "events_valid",
        "operator_counts",
        "policy_violation_count",
        "infrastructure_valid",
    ],
)
@dataclass
class EcosystemChunkMetrics:
    steps: jax.Array
    cumulative_reward: jax.Array
    birth_count: jax.Array
    death_count: jax.Array
    minimum_alive: jax.Array
    maximum_alive: jax.Array
    maximum_generation: jax.Array
    steps_at_capacity: jax.Array
    maximum_individual_id: jax.Array
    minimum_resource: jax.Array
    maximum_resource_excess: jax.Array
    minimum_live_energy: jax.Array
    finite: jax.Array
    identity_valid: jax.Array
    events_valid: jax.Array
    operator_counts: jax.Array
    policy_violation_count: jax.Array
    infrastructure_valid: jax.Array


def _state_numeric_valid(state: EcosystemState) -> jax.Array:
    """Validate finite simulation state while preserving legal NaN gene padding."""
    population = state.population
    strictly_finite = (
        state.nodes,
        state.edges,
        state.fields,
        population.alive,
        population.energy,
        population.age,
        population.generation,
        population.individual_id,
        population.parent_id,
        population.controller_order,
        population.controller_connection_index,
        population.founder_lineage_id,
        population.intake_ema,
        population.population_change_ema,
        population.next_individual_id,
        state.time,
        state.base_rest_lengths,
        state.base_bending_rest_angles,
        state.actuator_gain,
        state.resource_capacity_map,
        state.resource_regeneration_map,
    )
    leaves = jax.tree.leaves(strictly_finite)
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves]))
    order = population.controller_order
    connection_index = population.controller_connection_index
    cache_valid = jnp.all((order == I_INF) | ((order >= 0) & (order < MAX_NODES))) & jnp.all(
        (connection_index == I_INF) | ((connection_index >= 0) & (connection_index < MAX_CONNECTIONS))
    )
    actuator_valid = jnp.all((state.actuator_gain >= 0.0) & (state.actuator_gain <= 1.0))
    return finite & actuator_valid & cache_valid & population_numeric_valid(population.genome)


def _identity_valid(previous: EcosystemState, state: EcosystemState) -> jax.Array:
    population = state.population
    alive = population.alive
    sentinel = jnp.iinfo(jnp.int32).max
    live_ids = jnp.where(alive, population.individual_id, sentinel)
    sorted_ids = jnp.sort(live_ids)
    duplicated = jnp.any((sorted_ids[:-1] == sorted_ids[1:]) & (sorted_ids[:-1] != sentinel))
    ids_in_range = jnp.all(
        jnp.where(
            alive,
            (population.individual_id >= 0) & (population.individual_id < population.next_individual_id),
            True,
        )
    )
    return ~duplicated & ids_in_range & (population.next_individual_id >= previous.population.next_individual_id)


def initialize_chunk_metrics(state: EcosystemState) -> EcosystemChunkMetrics:
    alive_count = jnp.sum(state.population.alive).astype(jnp.int32)
    live_energy = jnp.min(jnp.where(state.population.alive, state.population.energy, jnp.inf))
    live_energy = jnp.where(jnp.any(state.population.alive), live_energy, 0.0)
    return EcosystemChunkMetrics(
        steps=jnp.array(0, dtype=jnp.int32),
        cumulative_reward=jnp.zeros((), dtype=state.fields.energy.dtype),
        birth_count=jnp.array(0, dtype=jnp.int32),
        death_count=jnp.array(0, dtype=jnp.int32),
        minimum_alive=alive_count,
        maximum_alive=alive_count,
        maximum_generation=jnp.max(state.population.generation),
        steps_at_capacity=jnp.array(0, dtype=jnp.int32),
        maximum_individual_id=jnp.max(state.population.individual_id),
        minimum_resource=jnp.min(state.fields.energy),
        maximum_resource_excess=jnp.max(state.fields.energy - state.resource_capacity_map),
        minimum_live_energy=live_energy,
        finite=_state_numeric_valid(state),
        identity_valid=jnp.array(True),
        events_valid=jnp.array(True),
        operator_counts=jnp.zeros(NUM_OPERATORS, dtype=jnp.int32),
        policy_violation_count=jnp.array(0, dtype=jnp.int32),
        infrastructure_valid=jnp.array(True),
    )


def update_chunk_metrics(
    metrics: EcosystemChunkMetrics,
    previous: EcosystemState,
    state: EcosystemState,
    reward: jax.Array,
    info: dict,
) -> EcosystemChunkMetrics:
    telemetry = info["telemetry"]
    alive_count = telemetry.alive_count
    any_alive = jnp.any(state.population.alive)
    step_live_energy = jnp.min(jnp.where(state.population.alive, state.population.energy, jnp.inf))
    minimum_live_energy = jnp.where(
        any_alive,
        jnp.minimum(metrics.minimum_live_energy, step_live_energy),
        metrics.minimum_live_energy,
    )
    births_valid = (telemetry.birth_count == jnp.sum(telemetry.birth_child_ids >= 0)) & (telemetry.birth_count == jnp.sum(telemetry.birth_parent_ids >= 0))
    deaths_valid = telemetry.death_count == jnp.sum(telemetry.death_ids >= 0)
    return EcosystemChunkMetrics(
        steps=metrics.steps + 1,
        cumulative_reward=metrics.cumulative_reward + reward,
        birth_count=metrics.birth_count + telemetry.birth_count,
        death_count=metrics.death_count + telemetry.death_count,
        minimum_alive=jnp.minimum(metrics.minimum_alive, alive_count),
        maximum_alive=jnp.maximum(metrics.maximum_alive, alive_count),
        maximum_generation=jnp.maximum(
            metrics.maximum_generation,
            jnp.max(state.population.generation),
        ),
        steps_at_capacity=(metrics.steps_at_capacity + (alive_count == state.population.alive.shape[0]).astype(jnp.int32)),
        maximum_individual_id=jnp.maximum(metrics.maximum_individual_id, jnp.max(state.population.individual_id)),
        minimum_resource=jnp.minimum(metrics.minimum_resource, jnp.min(state.fields.energy)),
        maximum_resource_excess=jnp.maximum(
            metrics.maximum_resource_excess,
            jnp.max(state.fields.energy - state.resource_capacity_map),
        ),
        minimum_live_energy=minimum_live_energy,
        finite=metrics.finite & _state_numeric_valid(state),
        identity_valid=metrics.identity_valid & _identity_valid(previous, state),
        events_valid=metrics.events_valid & births_valid & deaths_valid,
        operator_counts=metrics.operator_counts + telemetry.operator_counts,
        policy_violation_count=(metrics.policy_violation_count + telemetry.policy_violation_count),
        infrastructure_valid=(metrics.infrastructure_valid & telemetry.infrastructure_valid),
    )


def combine_chunk_metrics(
    first: EcosystemChunkMetrics,
    second: EcosystemChunkMetrics,
) -> EcosystemChunkMetrics:
    """Combine adjacent compact summaries without retaining their trajectories."""
    return EcosystemChunkMetrics(
        steps=first.steps + second.steps,
        cumulative_reward=first.cumulative_reward + second.cumulative_reward,
        birth_count=first.birth_count + second.birth_count,
        death_count=first.death_count + second.death_count,
        minimum_alive=jnp.minimum(first.minimum_alive, second.minimum_alive),
        maximum_alive=jnp.maximum(first.maximum_alive, second.maximum_alive),
        maximum_generation=jnp.maximum(first.maximum_generation, second.maximum_generation),
        steps_at_capacity=first.steps_at_capacity + second.steps_at_capacity,
        maximum_individual_id=jnp.maximum(first.maximum_individual_id, second.maximum_individual_id),
        minimum_resource=jnp.minimum(first.minimum_resource, second.minimum_resource),
        maximum_resource_excess=jnp.maximum(first.maximum_resource_excess, second.maximum_resource_excess),
        minimum_live_energy=jnp.minimum(first.minimum_live_energy, second.minimum_live_energy),
        finite=first.finite & second.finite,
        identity_valid=first.identity_valid & second.identity_valid,
        events_valid=first.events_valid & second.events_valid,
        operator_counts=first.operator_counts + second.operator_counts,
        policy_violation_count=(first.policy_violation_count + second.policy_violation_count),
        infrastructure_valid=(first.infrastructure_valid & second.infrastructure_valid),
    )


def run_ecosystem_chunk(env, state: EcosystemState, keys: jax.Array, action=None):
    """Return only final state and compact metrics for a fixed-size key chunk."""
    initial_metrics = initialize_chunk_metrics(state)

    def scan_step(carry, key):
        current, metrics = carry
        _, next_state, reward, _, info = env.step(key, current, action)
        metrics = update_chunk_metrics(metrics, current, next_state, reward, info)
        return (next_state, metrics), None

    (final_state, metrics), _ = jax.lax.scan(scan_step, (state, initial_metrics), keys)
    return final_state, metrics


def run_ecosystem_chunk_with_snapshots(
    env,
    state: EcosystemState,
    keys: jax.Array,
    snapshot_stride: int,
    action=None,
):
    """Explicit debug mode retaining one renderable snapshot per nested chunk."""
    if not isinstance(snapshot_stride, int) or snapshot_stride < 1:
        raise ValueError("snapshot_stride must be a positive integer")
    if keys.shape[0] % snapshot_stride:
        raise ValueError("key count must be divisible by snapshot_stride")
    blocks = keys.reshape(-1, snapshot_stride, *keys.shape[1:])
    initial_metrics = initialize_chunk_metrics(state)

    def block_step(carry, block_keys):
        current, accumulated = carry
        next_state, block_metrics = run_ecosystem_chunk(env, current, block_keys, action)
        accumulated = combine_chunk_metrics(accumulated, block_metrics)
        snapshot_fields = replace(next_state.fields, f_grid=None)
        snapshot = (
            next_state.nodes,
            snapshot_fields,
            next_state.population,
            next_state.time,
        )
        return (next_state, accumulated), snapshot

    (final_state, metrics), snapshots = jax.lax.scan(block_step, (state, initial_metrics), blocks)
    return final_state, metrics, snapshots
