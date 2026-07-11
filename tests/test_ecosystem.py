from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from microcosmos.gym import EcosystemEnv, EcosystemState, LineTopology, make
from microcosmos.rendering import render_fields
from microcosmos.solver.config import PBD_SCHEME_NO_FLUID
from microcosmos.utils import displacement


def _env(**kwargs):
    defaults = dict(
        topology=LineTopology(num_nodes=4),
        max_creatures=4,
        initial_population=2,
        grid_shape=(24, 24),
        solver_config=PBD_SCHEME_NO_FLUID,
        resource_regeneration_rate=0.0,
        mutation_std=0.0,
    )
    defaults.update(kwargs)
    return EcosystemEnv(**defaults)


def test_registry_builds_ecosystem():
    env = make(
        "ecosystem",
        topology=LineTopology(num_nodes=4),
        max_creatures=2,
        initial_population=1,
        grid_shape=(16, 16),
    )
    assert isinstance(env, EcosystemEnv)


def test_inherited_action_api_remains_compatible():
    env = _env()
    action = env.zero_action()
    assert env.action_spec() == {
        "d_rest_length": (env.num_edges,),
        "d_bending_angle": (env.num_bending_pairs,),
    }
    _, state = env.reset(jax.random.PRNGKey(0))
    implicit = env.step(jax.random.PRNGKey(1), state)
    explicit = env.step(jax.random.PRNGKey(1), state, action)
    assert jnp.array_equal(implicit[1].nodes.position, explicit[1].nodes.position)
    assert jnp.array_equal(implicit[1].population.energy, explicit[1].population.energy)


def test_reset_is_deterministic_and_uses_capacity_shapes():
    env = _env()
    obs_a, state_a = env.reset(jax.random.PRNGKey(0))
    obs_b, state_b = env.reset(jax.random.PRNGKey(0))
    assert isinstance(state_a, EcosystemState)
    assert int(jnp.sum(state_a.population.alive)) == env.initial_population
    assert state_a.nodes.position.shape == (env.max_creatures * 4, 2)
    assert state_a.population.genome.shape[0] == env.max_creatures
    assert jnp.array_equal(obs_a, obs_b)
    assert jnp.array_equal(state_a.nodes.position, state_b.nodes.position)
    centers = env._slot_centers(state_a.nodes.position)
    live_centers = centers[state_a.population.alive]
    center_distance = jnp.linalg.norm(
        displacement(env.grid_shape, live_centers[0], live_centers[1])
    )
    assert float(center_distance) > 2.0 * env.body_radius


def test_spawn_separation_is_derived_from_body_geometry():
    env = _env(initial_population=1, position_margin=1.25)
    assert np.isclose(
        env.spawn_separation, 2.0 * env.body_radius + env.position_margin
    )


def test_center_selection_uses_periodic_occupied_distance():
    env = _env(initial_population=0)
    candidates = jnp.array([[0.5, 12.0], [12.0, 12.0]])
    occupied_centers = jnp.array(
        [[23.5, 12.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    occupied = jnp.array([True, False, False, False])
    selected = env._select_best_center(
        candidates, occupied_centers, occupied
    )
    assert jnp.array_equal(selected, candidates[1])


def test_birth_occurs_at_end_of_step_with_correct_lineage():
    env = _env(
        initial_population=1,
        maturity_age=0,
        initial_energy=5.0,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        offspring_initial_energy=1.0,
        basal_metabolism=0.0,
        uptake_rate=0.0,
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    _, state, _, _, info = env.step(jax.random.PRNGKey(1), state)
    assert int(info["birth_count"]) == 1
    assert int(info["alive_count"]) == 2
    child_slot = int(info["telemetry"].birth_child_slots[0])
    assert int(state.population.parent_id[child_slot]) == 0
    assert int(state.population.generation[child_slot]) == 1
    assert float(state.population.energy[child_slot]) == 1.0
    centers = env._slot_centers(state.nodes.position)
    parent_distance = jnp.linalg.norm(
        displacement(env.grid_shape, centers[child_slot], centers[0])
    )
    assert np.isclose(float(parent_distance), env.spawn_separation, atol=1e-4)


def test_simultaneous_births_reserve_distinct_centers_and_lineage():
    env = _env(
        initial_population=2,
        maturity_age=0,
        initial_energy=5.0,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        offspring_initial_energy=1.0,
        basal_metabolism=0.0,
        uptake_rate=0.0,
    )
    _, initial = env.reset(jax.random.PRNGKey(4))
    _, state, _, _, info = env.step(jax.random.PRNGKey(5), initial)

    assert int(info["birth_count"]) == 2
    child_slots = np.asarray(info["telemetry"].birth_child_slots[:2])
    assert len(np.unique(child_slots)) == 2
    centers = env._slot_centers(state.nodes.position)
    child_distance = jnp.linalg.norm(
        displacement(
            env.grid_shape, centers[child_slots[0]], centers[child_slots[1]]
        )
    )
    assert float(child_distance) > 0.0
    assert np.array_equal(
        np.asarray(state.population.parent_id)[child_slots], np.array([0, 1])
    )


def test_dying_parent_cannot_reproduce_and_body_becomes_inert():
    env = _env(
        initial_population=1,
        maturity_age=0,
        initial_energy=0.1,
        reproduction_threshold=0.05,
        reproduction_cost=0.01,
        basal_metabolism=1.0,
        uptake_rate=0.0,
        dt=1.0,
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    _, state, _, _, info = env.step(jax.random.PRNGKey(1), state)
    assert int(info["death_count"]) == 1
    assert int(info["birth_count"]) == 0
    assert not jnp.any(state.nodes.active)
    assert jnp.all(state.nodes.velocity == 0.0)
    _, state2, _, _, _ = env.step(jax.random.PRNGKey(2), state)
    assert jnp.allclose(state2.nodes.position, state.nodes.position)


def test_jitted_steps_preserve_shapes_and_seeded_rollouts_are_identical():
    env = _env(max_steps=3)
    _, initial = env.reset(jax.random.PRNGKey(9))
    jit_step = jax.jit(env.step)
    state_a = initial
    state_b = initial
    for index in range(3):
        key = jax.random.PRNGKey(index)
        _, state_a, _, _, info_a = jit_step(key, state_a)
        _, state_b, _, _, info_b = jit_step(key, state_b)
        assert state_a.nodes.position.shape == initial.nodes.position.shape
        assert jnp.array_equal(info_a["birth_child_ids"], info_b["birth_child_ids"])
        assert jnp.array_equal(info_a["death_ids"], info_b["death_ids"])
    assert jnp.array_equal(state_a.population.individual_id, state_b.population.individual_id)
    assert jnp.all(jnp.isfinite(state_a.population.energy))
    assert jnp.all(jnp.isfinite(state_a.fields.energy))


def test_one_compiled_step_accepts_empty_partial_and_full_occupancy():
    env = _env()
    _, partial = env.reset(jax.random.PRNGKey(6))
    jit_step = jax.jit(env.step)
    states = []
    for alive in (
        jnp.zeros(env.max_creatures, dtype=jnp.bool_),
        partial.population.alive,
        jnp.ones(env.max_creatures, dtype=jnp.bool_),
    ):
        population = replace(partial.population, alive=alive)
        nodes = replace(partial.nodes, active=alive[env.node_slot])
        state = replace(partial, population=population, nodes=nodes)
        _, result, _, _, _ = jit_step(jax.random.PRNGKey(7), state)
        states.append(result)
    assert all(state.nodes.position.shape == partial.nodes.position.shape for state in states)
    assert all(state.population.genome.shape == partial.population.genome.shape for state in states)


def test_render_omits_inactive_slots():
    env = _env(initial_population=1)
    _, state = env.reset(jax.random.PRNGKey(0))
    positions_a = jnp.where(
        state.nodes.active[:, None], state.nodes.position, 0.0
    )
    positions_b = jnp.where(
        state.nodes.active[:, None], state.nodes.position, 12.0
    )
    frame_a = env.render(
        replace(state, nodes=replace(state.nodes, position=positions_a)), size=64
    )
    frame_b = env.render(
        replace(state, nodes=replace(state.nodes, position=positions_b)), size=64
    )
    assert frame_a.shape == (64, 64, 3)
    assert frame_a.dtype == np.uint8
    assert np.array_equal(frame_a, frame_b)


def test_environment_render_is_exactly_the_native_field_renderer():
    env = _env(initial_population=1)
    _, state = env.reset(jax.random.PRNGKey(0))
    nodes_ts = jax.tree.map(lambda value: value[None], state.nodes)
    fields_ts = replace(
        state.fields,
        steric=state.fields.steric[None],
        fluid_velocity=state.fields.fluid_velocity[None],
        f_grid=None,
        energy=state.fields.energy[None],
    )
    direct = render_fields(
        fields_ts,
        nodes_ts,
        sz=64,
        animate_energy=False,
    )[0]
    assert np.array_equal(env.render(state, size=64), direct)


def test_birth_remains_finite_when_ideal_clearance_is_impossible():
    env = _env(
        grid_shape=(8, 8),
        initial_population=1,
        maturity_age=0,
        initial_energy=5.0,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        basal_metabolism=0.0,
        uptake_rate=0.0,
    )
    _, initial = env.reset(jax.random.PRNGKey(10))
    _, state, _, _, info = jax.jit(env.step)(jax.random.PRNGKey(11), initial)
    assert int(info["birth_count"]) == 1
    assert jnp.all(jnp.isfinite(state.nodes.position))
    assert jnp.all(jnp.isfinite(state.population.energy))
