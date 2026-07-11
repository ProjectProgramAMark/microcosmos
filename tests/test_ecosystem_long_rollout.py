from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microcosmos.gym import EcosystemEnv, LineTopology
from microcosmos.rollout import initialize_chunk_metrics, update_chunk_metrics
from microcosmos.solver.config import PBD_SCHEME


def test_guarded_fluid_forced_turnover_rollout():
    if jax.default_backend() != "gpu":
        pytest.skip("the 1,000-step fluid regression runs only in the GPU gate")

    horizon = 1_000
    turnover_interval = 100
    maximum_lifespan = 10_000
    solver = replace(
        PBD_SCHEME,
        cycles_per_step=1,
        ibm_iterations=1,
        ibm_kernel_size=2,
        synthetic_node_width=1.0,
    )
    env = EcosystemEnv(
        topology=LineTopology(num_nodes=4),
        max_creatures=4,
        initial_population=2,
        grid_shape=(24, 24),
        dt=0.01,
        max_steps=horizon,
        solver_config=solver,
        initial_resource=0.0,
        resource_regeneration_rate=0.0,
        initial_energy=2.0,
        maturity_age=0,
        maximum_lifespan=maximum_lifespan,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        birth_transfer_efficiency=0.5,
        uptake_rate=0.0,
        basal_metabolism=0.0,
        actuation_power_coefficient=0.0,
        mutation_probability=1.0,
        mutation_std=0.1,
    )
    _, initial = env.reset(jax.random.PRNGKey(100))
    initial_inactive = ~initial.nodes.active
    initial_inactive_positions = initial.nodes.position[initial_inactive]
    keys = jax.random.split(jax.random.PRNGKey(101), horizon)

    def rollout(state, scan_keys):
        initial_metrics = initialize_chunk_metrics(state)

        def scan_step(carry, key):
            current, metrics = carry
            force_turnover = current.time % turnover_interval == 0
            forced_energy = current.population.energy.at[1].set(5.0)
            forced_age = current.population.age.at[0].set(maximum_lifespan - 1)
            population = replace(
                current.population,
                energy=jnp.where(
                    force_turnover, forced_energy, current.population.energy
                ),
                age=jnp.where(force_turnover, forced_age, current.population.age),
            )
            current = replace(current, population=population)
            _, next_state, reward, _, info = env.step(key, current)
            metrics = update_chunk_metrics(
                metrics, current, next_state, reward, info
            )
            telemetry = info["telemetry"]
            events = (
                telemetry.birth_count,
                telemetry.death_count,
                telemetry.birth_child_slots[0],
                telemetry.death_ids[0],
                telemetry.birth_child_ids[0],
            )
            return (next_state, metrics), events

        return jax.lax.scan(scan_step, (state, initial_metrics), scan_keys)

    (final, metrics), events = jax.jit(rollout)(initial, keys)
    jax.block_until_ready(final.nodes.position)
    births, deaths, child_slots, death_ids, child_ids = events
    forced_steps = jnp.arange(horizon) % turnover_interval == 0

    assert int(metrics.steps) == horizon
    assert int(metrics.birth_count) == int(jnp.sum(forced_steps)) == 10
    assert int(metrics.death_count) == 10
    assert bool(metrics.finite)
    assert bool(metrics.identity_valid)
    assert bool(metrics.events_valid)
    assert metrics.minimum_resource >= -1e-6
    assert metrics.maximum_resource_excess <= 1e-6
    assert metrics.minimum_live_energy > 0.0
    assert jnp.all(births[forced_steps] == 1)
    assert jnp.all(deaths[forced_steps] == 1)
    assert jnp.all(child_slots[forced_steps] == 0)
    assert jnp.all(death_ids[forced_steps] >= 0)
    assert jnp.array_equal(
        child_ids[forced_steps], jnp.arange(2, 12, dtype=jnp.int32)
    )
    assert int(final.population.next_individual_id) == 12
    assert int(jnp.sum(final.population.alive)) == 2
    assert jnp.array_equal(final.nodes.active, final.population.alive[env.node_slot])
    assert jnp.allclose(
        final.nodes.position[initial_inactive],
        initial_inactive_positions,
        atol=1e-7,
    )
    assert jnp.all(jnp.isfinite(final.population.energy))
    assert jnp.all(final.fields.energy >= -1e-6)
    assert jnp.all(
        final.fields.energy <= final.resource_capacity_map + 1e-6
    )
    assert final.nodes.position.shape == initial.nodes.position.shape
    assert final.population.genome.shape == initial.population.genome.shape

    moved_inactive = replace(
        final.nodes,
        position=jnp.where(
            final.nodes.active[:, None], final.nodes.position, 1.0
        ),
    )
    frame = env.render(final, size=64)
    moved_frame = env.render(replace(final, nodes=moved_inactive), size=64)
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8
    assert np.array_equal(frame, moved_frame)
