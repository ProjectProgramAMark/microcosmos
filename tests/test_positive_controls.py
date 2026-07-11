import jax
import jax.numpy as jnp

from microcosmos.positive_controls import (
    foraging_control_environment,
    foraging_gate_metrics,
    movement_control_environment,
    movement_gate_metrics,
    sensory_foraging_genome,
    sensor_disabled_genome,
    traveling_wave_genome,
    with_relative_resource_patch,
    zero_action_genome,
)
from microcosmos.utils import displacement


def test_control_genomes_are_normalized_and_differ_only_as_declared():
    zero = zero_action_genome()
    wave = traveling_wave_genome()
    sensory = sensory_foraging_genome()
    disabled = sensor_disabled_genome()
    for genome in (zero, wave, sensory, disabled):
        assert genome.shape == (23,)
        assert genome.dtype == jnp.float32
        assert jnp.all(jnp.abs(genome) <= 1.0)
    assert jnp.all(zero == 0.0)
    assert jnp.array_equal(sensory[:12], disabled[:12])
    assert jnp.array_equal(sensory[18:], disabled[18:])
    assert jnp.any(sensory[12:18] != disabled[12:18])
    assert jnp.all(disabled[12:18] == 0.0)


def test_positive_control_environments_use_the_scientific_fluid_fixture():
    movement = movement_control_environment(horizon=10)
    foraging = foraging_control_environment(horizon=10)
    assert movement.solver_config.enable_fluid
    assert foraging.solver_config.enable_fluid
    assert movement._nodes_per_slot == foraging._nodes_per_slot == 8
    assert movement.dt == foraging.dt == 0.01
    assert foraging.genome_config.resource_reference == 0.0


def test_foraging_patch_starts_outside_the_body_and_is_map_bounded():
    env = foraging_control_environment(horizon=10)
    _, state = env.reset(jax.random.PRNGKey(0))
    patched = with_relative_resource_patch(env, state)
    body_center = env._slot_centers(state.nodes.position)[0]
    patch_center = body_center + jnp.array([-11.0, 0.0])
    node_distance = jnp.linalg.norm(
        displacement(env.grid_shape, state.nodes.position, patch_center), axis=-1
    )
    assert jnp.min(node_distance) > 3.0
    assert jnp.array_equal(
        patched.fields.energy, patched.resource_capacity_map
    )
    assert jnp.all(patched.resource_capacity_map >= 0.0)
    assert jnp.all(
        patched.resource_capacity_map <= env.resource_config.capacity + 1e-6
    )


def test_positive_control_gate_metrics_enforce_all_thresholds():
    movement = movement_gate_metrics(
        jnp.array([1.5, 0.0]), jnp.array([0.1, 0.0]), body_length=14.0
    )
    assert bool(movement["passed"])
    assert movement["body_fraction"] >= 0.1
    assert movement["drift_ratio"] >= 5.0

    foraging = foraging_gate_metrics(
        jnp.array([1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 0.9, 0.9]),
        jnp.ones(8),
    )
    assert bool(foraging["passed"])
    assert foraging["wins"] == 6
