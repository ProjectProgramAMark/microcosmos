from dataclasses import replace

import jax
import jax.numpy as jnp

from microcosmos.forces import update_steric_potential
from microcosmos.gym import EcosystemEnv, LineTopology
from microcosmos.simulate import step
from microcosmos.solver.config import PBD_SCHEME_NO_FLUID
from microcosmos.solver.masks import (
    activity_masks,
    bending_activity,
    edge_activity,
    node_activity,
)


def _environment(enable_fluid=False, synthetic_node_width=0.0):
    solver = replace(
        PBD_SCHEME_NO_FLUID,
        enable_fluid=enable_fluid,
        cycles_per_step=1,
        ibm_iterations=1,
        ibm_kernel_size=2,
        synthetic_node_width=synthetic_node_width,
    )
    return EcosystemEnv(
        topology=LineTopology(num_nodes=4),
        max_creatures=2,
        initial_population=1,
        grid_shape=(24, 24),
        solver_config=solver,
    )


def test_activity_masks_have_static_expected_shapes():
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(0))
    assert node_activity(state.nodes).shape == (env.num_nodes,)
    assert edge_activity(state.nodes, state.edges).shape == (env.num_edges,)
    assert bending_activity(state.nodes, state.edges).shape == (env.num_bending_pairs,)
    masks = activity_masks(state.nodes, state.edges)
    assert masks[0].shape == (env.num_nodes,)
    assert masks[1].shape == (env.num_edges,)
    assert masks[2].shape == (env.num_bending_pairs,)
    assert jnp.all(masks[1][env.edge_slices[0]])
    assert not jnp.any(masks[1][env.edge_slices[1]])


def test_inactive_nodes_remain_stationary_with_nonzero_input_velocity():
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(0))
    inactive = ~state.nodes.active
    nodes = replace(state.nodes, velocity=jnp.where(inactive[:, None], 10.0, state.nodes.velocity))
    before = nodes.position
    nodes, _, _ = step(nodes, state.edges, state.fields, env.dt, env.solver_config)
    assert jnp.allclose(nodes.position[inactive], before[inactive], atol=1e-7)
    assert jnp.all(nodes.velocity[inactive] == 0.0)


def test_inactive_positions_do_not_change_active_no_fluid_physics():
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(1))
    inactive = ~state.nodes.active
    moved = replace(state.nodes, position=jnp.where(inactive[:, None], 3.0, state.nodes.position))
    result_a, _, _ = step(state.nodes, state.edges, state.fields, env.dt, env.solver_config)
    result_b, _, _ = step(moved, state.edges, state.fields, env.dt, env.solver_config)
    assert jnp.allclose(result_a.position[state.nodes.active], result_b.position[state.nodes.active], atol=1e-7)


def test_inactive_nodes_deposit_no_steric_mass():
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(2))
    active_only = state.nodes[state.nodes.active]
    full = update_steric_potential(state.nodes, state.fields, sigma=1.5)
    reference = update_steric_potential(active_only, state.fields, sigma=1.5)
    assert jnp.allclose(full.steric, reference.steric, atol=1e-7)


def test_inactive_positions_do_not_change_fluid_field():
    env = _environment(enable_fluid=True)
    _, state = env.reset(jax.random.PRNGKey(3))
    inactive = ~state.nodes.active
    moved = replace(state.nodes, position=jnp.where(inactive[:, None], 1.0, state.nodes.position))
    _, _, fields_a = step(state.nodes, state.edges, state.fields, env.dt, env.solver_config)
    _, _, fields_b = step(moved, state.edges, state.fields, env.dt, env.solver_config)
    assert jnp.allclose(fields_a.fluid_velocity, fields_b.fluid_velocity, atol=1e-7)
    assert jnp.allclose(fields_a.f_grid, fields_b.f_grid, atol=1e-7)


def test_inactive_incident_edge_does_not_change_synthetic_fluid_tangent():
    env = _environment(enable_fluid=True, synthetic_node_width=1.0)
    _, state = env.reset(jax.random.PRNGKey(30))
    active = state.nodes.active.at[env.node_slices[0].stop - 1].set(False)
    inactive_endpoint = env.node_slices[0].stop - 1
    nodes_a = replace(state.nodes, active=active)
    nodes_b = replace(
        nodes_a,
        position=nodes_a.position.at[inactive_endpoint].add(jnp.array([5.0, 7.0])),
    )
    result_a, _, fields_a = step(
        nodes_a, state.edges, state.fields, env.dt, env.solver_config
    )
    result_b, _, fields_b = step(
        nodes_b, state.edges, state.fields, env.dt, env.solver_config
    )
    active_reaction_difference = jnp.max(
        jnp.abs(result_a.velocity[active] - result_b.velocity[active])
    )
    assert active_reaction_difference <= 1e-7
    assert jnp.allclose(fields_a.fluid_velocity, fields_b.fluid_velocity, atol=1e-7)
    assert jnp.allclose(fields_a.f_grid, fields_b.f_grid, atol=1e-7)


def test_ecosystem_population_activity_is_authoritative_at_step_entry():
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(31))
    inconsistent = replace(
        state,
        nodes=replace(state.nodes, active=~state.nodes.active),
    )
    key = jax.random.PRNGKey(32)
    _, canonical_result, _, _, _ = env.step(key, state)
    _, repaired_result, _, _, _ = env.step(key, inconsistent)
    assert jnp.array_equal(
        repaired_result.nodes.active,
        repaired_result.population.alive[env.node_slot],
    )
    assert jnp.allclose(
        canonical_result.nodes.position, repaired_result.nodes.position, atol=1e-7
    )


def test_masked_continuous_physics_has_finite_gradients():
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(4))

    def objective(position):
        nodes, _, _ = step(
            replace(state.nodes, position=position),
            state.edges,
            state.fields,
            env.dt,
            env.solver_config,
        )
        return jnp.sum(nodes.position[state.nodes.active] ** 2)

    gradient = jax.grad(objective)(state.nodes.position)
    assert jnp.all(jnp.isfinite(gradient))
