"""Deterministic paired controls for embodied movement and acquisition."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from microcosmos.cppn import (
    CONNECTION_WEIGHT,
    CPPNGenome,
    canonical_cppn_genome,
    transform_population,
)
from microcosmos.gym.ecosystem import EcosystemEnv
from microcosmos.gym.multi_agent import LineTopology
from microcosmos.ecology import make_periodic_resource_patch
from microcosmos.solver.config import PBD_SCHEME
from microcosmos.utils import displacement


def zero_action_genome() -> CPPNGenome:
    """Return a valid CPPN whose decoded action is identically zero."""
    genome = canonical_cppn_genome()
    connections = genome.connection_genes.at[:, CONNECTION_WEIGHT].set(
        jnp.where(
            jnp.isnan(genome.connection_genes[:, CONNECTION_WEIGHT]),
            jnp.nan,
            0.0,
        )
    )
    nodes = genome.node_genes.at[:, 1].set(jnp.where(jnp.isnan(genome.node_genes[:, 1]), jnp.nan, 0.0))
    return CPPNGenome(nodes, connections)


def traveling_wave_genome() -> CPPNGenome:
    """Return the frozen single-wave movement positive control."""
    genome = canonical_cppn_genome()
    return CPPNGenome(
        genome.node_genes,
        genome.connection_genes.at[3:5, CONNECTION_WEIGHT].set(0.0),
    )


def sensory_foraging_genome() -> CPPNGenome:
    """Return a hand-set wave plus resource corrections for paired foraging."""
    return canonical_cppn_genome()


def sensor_disabled_genome() -> CPPNGenome:
    """Match the foraging CPPN while zeroing only its two sensor weights."""
    return traveling_wave_genome()


def movement_control_environment(horizon: int = 5_000) -> EcosystemEnv:
    """Construct the frozen fluid-on locomotion fixture."""
    return EcosystemEnv(
        topology=LineTopology(num_nodes=8, spacing=2.0),
        max_creatures=1,
        initial_population=1,
        grid_shape=(64, 64),
        dt=0.01,
        max_steps=horizon,
        solver_config=PBD_SCHEME,
        initial_resource=0.0,
        resource_regeneration_rate=0.0,
        initial_energy=100.0,
        reproduction_threshold=1_000_000.0,
        reproduction_cost=1.0,
        maturity_age=horizon + 1,
        maximum_lifespan=horizon + 100,
        uptake_rate=0.0,
        basal_metabolism=0.0,
        actuation_power_coefficient=0.0,
        placement_candidates=1,
    )


def foraging_control_environment(horizon: int = 5_000) -> EcosystemEnv:
    """Construct the paired localized-resource acquisition fixture."""
    return EcosystemEnv(
        topology=LineTopology(num_nodes=8, spacing=2.0),
        max_creatures=1,
        initial_population=1,
        grid_shape=(64, 64),
        dt=0.01,
        max_steps=horizon,
        solver_config=PBD_SCHEME,
        resource_capacity=1.0,
        initial_resource=0.0,
        resource_regeneration_rate=0.0,
        initial_energy=100.0,
        reproduction_threshold=1_000_000.0,
        reproduction_cost=1.0,
        maturity_age=horizon + 1,
        maximum_lifespan=horizon + 100,
        uptake_rate=1.0,
        basal_metabolism=0.0,
        actuation_power_coefficient=0.0,
        placement_candidates=1,
    )


def with_relative_resource_patch(
    env: EcosystemEnv,
    state,
    offset: tuple[float, float] = (-11.5, 0.0),
    radius: float = 4.0,
):
    """Place a full compact patch at a declared offset from the body center."""
    center = env._slot_centers(state.nodes.position)[0] + jnp.asarray(offset)
    capacity, regeneration = make_periodic_resource_patch(
        env.grid_shape,
        center,
        radius,
        env.resource_config.capacity,
        env.resource_config.regeneration_rate,
    )
    return replace(
        state,
        fields=replace(state.fields, energy=capacity),
        resource_capacity_map=capacity,
        resource_regeneration_map=regeneration,
    )


def with_genome(state, genome: CPPNGenome):
    """Replace the sole live organism's genome without changing world state."""
    population_genome = jax.tree.map(lambda value: value[None, ...], genome)
    order, connection_index = transform_population(population_genome)
    return replace(
        state,
        population=replace(
            state.population,
            genome=population_genome,
            controller_order=order,
            controller_connection_index=connection_index,
        ),
    )


def paired_rollout(
    env: EcosystemEnv,
    state_a,
    state_b,
    keys: jax.Array,
):
    """Run two policies through paired keys and return raw compact traces."""
    zero_vector = jnp.zeros(2, dtype=state_a.nodes.position.dtype)
    zero_total = jnp.zeros((), dtype=state_a.nodes.position.dtype)

    def scan_step(carry, key):
        current_a, current_b, unwrapped_a, unwrapped_b, uptake_a, uptake_b = carry
        center_a = env._slot_centers(current_a.nodes.position)[0]
        center_b = env._slot_centers(current_b.nodes.position)[0]
        _, next_a, reward_a, _, _ = env.step(key, current_a)
        _, next_b, reward_b, _, _ = env.step(key, current_b)
        delta_a = displacement(env.grid_shape, env._slot_centers(next_a.nodes.position)[0], center_a)
        delta_b = displacement(env.grid_shape, env._slot_centers(next_b.nodes.position)[0], center_b)
        unwrapped_a = unwrapped_a + delta_a
        unwrapped_b = unwrapped_b + delta_b
        uptake_a = uptake_a + reward_a
        uptake_b = uptake_b + reward_b
        return (
            next_a,
            next_b,
            unwrapped_a,
            unwrapped_b,
            uptake_a,
            uptake_b,
        ), (unwrapped_a, unwrapped_b, uptake_a, uptake_b)

    initial = (state_a, state_b, zero_vector, zero_vector, zero_total, zero_total)
    final, trace = jax.lax.scan(scan_step, initial, keys)
    return final, trace


def movement_gate_metrics(
    wave_displacement: jax.Array,
    zero_displacement: jax.Array,
    body_length: float,
) -> dict[str, jax.Array]:
    """Calculate the frozen locomotion gate metrics."""
    wave_distance = jnp.linalg.norm(wave_displacement)
    zero_distance = jnp.linalg.norm(zero_displacement)
    return {
        "wave_distance": wave_distance,
        "zero_distance": zero_distance,
        "body_fraction": wave_distance / body_length,
        "drift_ratio": wave_distance / jnp.maximum(zero_distance, 1e-12),
        "passed": (wave_distance >= 0.1 * body_length) & (wave_distance >= 5.0 * zero_distance),
    }


def foraging_gate_metrics(
    sensory_uptake: jax.Array,
    control_uptake: jax.Array,
) -> dict[str, jax.Array]:
    """Calculate the frozen eight-world paired acquisition metrics."""
    mean_control = jnp.mean(control_uptake)
    median_control = jnp.median(control_uptake)
    mean_improvement = (jnp.mean(sensory_uptake) - mean_control) / jnp.maximum(mean_control, 1e-12)
    median_improvement = (jnp.median(sensory_uptake) - median_control) / jnp.maximum(median_control, 1e-12)
    wins = jnp.sum(sensory_uptake > control_uptake)
    return {
        "mean_improvement": mean_improvement,
        "median_improvement": median_improvement,
        "wins": wins,
        "passed": (mean_improvement >= 0.2) & (median_improvement >= 0.2) & (wins >= 6),
    }
