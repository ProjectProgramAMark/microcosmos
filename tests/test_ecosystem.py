from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import microcosmos.gym.ecosystem as ecosystem_module
from microcosmos.cppn import (
    CPPNGenome,
    MAX_CONNECTIONS,
    MAX_NODES,
    cached_genome_valid,
    population_numeric_valid,
    transform_population,
)
from microcosmos.gym import (
    EcosystemEnv,
    EcosystemState,
    LineTopology,
    RingTopology,
    make,
)
from microcosmos.heredity import clone_policy
from microcosmos.positive_controls import (
    traveling_wave_genome,
    zero_action_genome,
)
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
        offspring_policy=clone_policy,
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


def test_environment_rejects_invalid_founder_panel(monkeypatch):
    transform = ecosystem_module.transform_and_validate_population

    def invalidate(genome):
        canonical, order, connection_index, valid = transform(genome)
        return canonical, order, connection_index, jnp.zeros_like(valid)

    monkeypatch.setattr(
        ecosystem_module,
        "transform_and_validate_population",
        invalidate,
    )
    with pytest.raises(RuntimeError, match="founder CPPN panel"):
        _env()


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
    _, state_other_world = env.reset(jax.random.PRNGKey(123))
    assert isinstance(state_a, EcosystemState)
    assert int(jnp.sum(state_a.population.alive)) == env.initial_population
    assert state_a.nodes.position.shape == (env.max_creatures * 4, 2)
    assert state_a.population.genome.node_genes.shape == (
        env.max_creatures,
        MAX_NODES,
        5,
    )
    assert state_a.population.genome.connection_genes.shape == (
        env.max_creatures,
        MAX_CONNECTIONS,
        3,
    )
    assert state_a.population.genome.node_genes.dtype == jnp.float32
    assert state_a.population.genome.connection_genes.dtype == jnp.float32
    assert bool(population_numeric_valid(state_a.population.genome))
    for slot in range(env.max_creatures):
        assert bool(
            cached_genome_valid(
                CPPNGenome(
                    state_a.population.genome.node_genes[slot],
                    state_a.population.genome.connection_genes[slot],
                ),
                state_a.population.controller_order[slot],
                state_a.population.controller_connection_index[slot],
            )
        )
    assert state_a.population.founder_lineage_id.tolist() == [0, 1, -1, -1]
    assert state_a.resource_capacity_map.shape == env.grid_shape
    assert state_a.resource_regeneration_map.shape == env.grid_shape
    assert jnp.all(state_a.fields.energy >= 0.0)
    assert jnp.all(state_a.fields.energy <= state_a.resource_capacity_map + 1e-6)
    assert jnp.all(state_a.fields.energy[state_a.resource_capacity_map == 0.0] == 0.0)
    assert jnp.array_equal(obs_a, obs_b)
    assert jnp.array_equal(state_a.nodes.position, state_b.nodes.position)
    assert jnp.array_equal(
        state_a.population.genome.node_genes,
        state_b.population.genome.node_genes,
        equal_nan=True,
    )
    assert jnp.array_equal(
        state_a.population.genome.connection_genes,
        state_b.population.genome.connection_genes,
        equal_nan=True,
    )
    # Founder genetics are a frozen paired panel, independent of world draws.
    assert jnp.array_equal(
        state_a.population.genome.node_genes,
        state_other_world.population.genome.node_genes,
        equal_nan=True,
    )
    assert jnp.array_equal(
        state_a.population.genome.connection_genes,
        state_other_world.population.genome.connection_genes,
        equal_nan=True,
    )
    assert not jnp.array_equal(state_a.nodes.position, state_other_world.nodes.position)
    centers = env._slot_centers(state_a.nodes.position)
    live_centers = centers[state_a.population.alive]
    center_distance = jnp.linalg.norm(displacement(env.grid_shape, live_centers[0], live_centers[1]))
    assert float(center_distance) > 2.0 * env.body_radius


def test_explicit_founder_initializes_clones_without_collapsing_pedigree_ids():
    founder = zero_action_genome()
    env = _env(
        max_creatures=4,
        initial_population=2,
        founder_genome=founder,
    )
    _, state = env.reset(jax.random.PRNGKey(1000))

    for slot in range(env.max_creatures):
        assert jnp.array_equal(
            state.population.genome.node_genes[slot],
            state.population.genome.node_genes[0],
            equal_nan=True,
        )
        assert jnp.array_equal(
            state.population.genome.connection_genes[slot],
            state.population.genome.connection_genes[0],
            equal_nan=True,
        )
    assert state.population.founder_lineage_id.tolist() == [0, 1, -1, -1]


def test_one_environment_can_reset_with_different_explicit_founders():
    env = _env(max_creatures=3, initial_population=2)
    reset_key = jax.random.PRNGKey(1001)

    _, zero_state = env.reset(reset_key, founder_genome=zero_action_genome())
    _, wave_state = env.reset(reset_key, founder_genome=traveling_wave_genome())

    assert zero_state.population.genome.node_genes.shape == wave_state.population.genome.node_genes.shape
    assert zero_state.population.genome.connection_genes.shape == wave_state.population.genome.connection_genes.shape
    assert not jnp.array_equal(
        zero_state.population.genome.connection_genes,
        wave_state.population.genome.connection_genes,
        equal_nan=True,
    )
    assert jnp.array_equal(zero_state.nodes.position, wave_state.nodes.position)
    assert zero_state.population.founder_lineage_id.tolist() == [0, 1, -1]
    assert wave_state.population.founder_lineage_id.tolist() == [0, 1, -1]

    compiled_step = jax.jit(env.step)
    zero_result = compiled_step(jax.random.PRNGKey(1002), zero_state)[1]
    wave_result = compiled_step(jax.random.PRNGKey(1002), wave_state)[1]
    assert zero_result.nodes.position.shape == wave_result.nodes.position.shape


def test_spawn_separation_is_derived_from_body_geometry():
    env = _env(initial_population=1, position_margin=1.25)
    assert np.isclose(env.spawn_separation, 2.0 * env.body_radius + env.position_margin)


def test_center_selection_uses_periodic_occupied_distance():
    env = _env(initial_population=0)
    candidates = jnp.array([[0.5, 12.0], [12.0, 12.0]])
    occupied_centers = jnp.array([[23.5, 12.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    occupied = jnp.array([True, False, False, False])
    selected = env._select_best_center(candidates, occupied_centers, occupied)
    assert jnp.array_equal(selected, candidates[1])


def test_birth_occurs_at_end_of_step_with_correct_lineage():
    env = _env(
        initial_population=1,
        maturity_age=0,
        initial_energy=5.0,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        birth_transfer_efficiency=0.5,
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
    assert int(state.population.founder_lineage_id[child_slot]) == 0
    assert float(state.population.energy[child_slot]) == 1.0
    child = CPPNGenome(
        state.population.genome.node_genes[child_slot],
        state.population.genome.connection_genes[child_slot],
    )
    assert bool(
        cached_genome_valid(
            child,
            state.population.controller_order[child_slot],
            state.population.controller_connection_index[child_slot],
        )
    )
    assert info["operator_counts"].tolist() == [1, 0, 0, 0]
    assert int(info["policy_violation_count"]) == 0
    assert bool(info["infrastructure_valid"])
    centers = env._slot_centers(state.nodes.position)
    parent_distance = jnp.linalg.norm(displacement(env.grid_shape, centers[child_slot], centers[0]))
    assert np.isclose(float(parent_distance), env.spawn_separation, atol=1e-4)


def test_simultaneous_births_reserve_distinct_centers_and_lineage():
    env = _env(
        initial_population=2,
        maturity_age=0,
        initial_energy=5.0,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        birth_transfer_efficiency=0.5,
        basal_metabolism=0.0,
        uptake_rate=0.0,
    )
    _, initial = env.reset(jax.random.PRNGKey(4))
    _, state, _, _, info = env.step(jax.random.PRNGKey(5), initial)

    assert int(info["birth_count"]) == 2
    child_slots = np.asarray(info["telemetry"].birth_child_slots[:2])
    assert len(np.unique(child_slots)) == 2
    centers = env._slot_centers(state.nodes.position)
    child_distance = jnp.linalg.norm(displacement(env.grid_shape, centers[child_slots[0]], centers[child_slots[1]]))
    assert float(child_distance) > 0.0
    assert np.array_equal(
        np.sort(np.asarray(state.population.parent_id)[child_slots]),
        np.array([0, 1]),
    )
    assert np.array_equal(
        np.sort(np.asarray(state.population.founder_lineage_id)[child_slots]),
        np.array([0, 1]),
    )
    assert info["operator_counts"].tolist() == [2, 0, 0, 0]


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
    assert all(state.population.genome.node_genes.shape == partial.population.genome.node_genes.shape for state in states)
    assert all(state.population.genome.connection_genes.shape == partial.population.genome.connection_genes.shape for state in states)


def test_genome_loci_cannot_change_fixed_ecological_traits():
    env = _env(max_bending_delta=0.0)
    _, initial = env.reset(jax.random.PRNGKey(40))

    def population_with_genome(genome):
        batched = jax.tree.map(
            lambda value: jnp.broadcast_to(value, (env.max_creatures, *value.shape)),
            genome,
        )
        order, connection_index = transform_population(batched)
        return replace(
            initial.population,
            genome=batched,
            controller_order=order,
            controller_connection_index=connection_index,
        )

    low = replace(
        initial,
        population=population_with_genome(zero_action_genome()),
    )
    high = replace(
        initial,
        population=population_with_genome(traveling_wave_genome()),
    )
    key = jax.random.PRNGKey(41)
    _, low_result, _, _, _ = env.step(key, low)
    _, high_result, _, _, _ = env.step(key, high)
    assert jnp.array_equal(low_result.fields.energy, high_result.fields.energy)
    assert jnp.array_equal(low_result.population.energy, high_result.population.energy)


def test_ecosystem_controller_rejects_unsupported_topology():
    with pytest.raises(ValueError, match="LineTopology"):
        _env(topology=RingTopology(num_nodes=4))


def test_render_omits_inactive_slots():
    env = _env(initial_population=1)
    _, state = env.reset(jax.random.PRNGKey(0))
    positions_a = jnp.where(state.nodes.active[:, None], state.nodes.position, 0.0)
    positions_b = jnp.where(state.nodes.active[:, None], state.nodes.position, 12.0)
    frame_a = env.render(replace(state, nodes=replace(state.nodes, position=positions_a)), size=64)
    frame_b = env.render(replace(state, nodes=replace(state.nodes, position=positions_b)), size=64)
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grid_shape": (0, 24)}, "grid_shape"),
        ({"dt": 0.0}, "dt"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_creatures": 0}, "max_creatures"),
        ({"initial_population": -1}, "initial_population"),
        ({"initial_resource": -0.1}, "initial_resource"),
        (
            {"resource_capacity": 1.0, "initial_resource": 1.1},
            "initial_resource",
        ),
        ({"initial_energy": -0.1}, "initial_energy"),
        ({"actuation_power_coefficient": -0.1}, "actuation_power_coefficient"),
        ({"assimilation_efficiency": 1.1}, "assimilation_efficiency"),
        ({"uptake_rate": -0.1}, "uptake_rate"),
        ({"basal_metabolism": -0.1}, "basal_metabolism"),
        ({"max_bending_delta": -0.1}, "max_bending_delta"),
        ({"resource_patch_center": (0.0, jnp.inf)}, "resource_patch_center"),
        ({"resource_patch_radius": 0.0}, "resource_patch_radius"),
        (
            {"dt": 1.0, "resource_diffusion_rate": 1.1},
            "resource_diffusion_rate",
        ),
        ({"placement_candidates": 0}, "placement_candidates"),
        ({"placement_candidates": 1.5}, "placement_candidates"),
        ({"placement_candidates": True}, "placement_candidates"),
        ({"spawn_separation": -1.0}, "spawn_separation"),
    ],
)
def test_environment_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _env(**kwargs)
