import jax
import jax.numpy as jnp

from microcosmos.cppn import (
    CONNECTION_WEIGHT,
    MAX_CONNECTIONS,
    MAX_NODES,
    NODE_BIAS,
    CPPNGenome,
    cached_genome_valid,
    genome_numeric_valid,
)
from microcosmos.positive_controls import (
    foraging_control_environment,
    foraging_gate_metrics,
    movement_control_environment,
    movement_gate_metrics,
    sensory_foraging_genome,
    sensor_disabled_genome,
    traveling_wave_genome,
    with_genome,
    with_relative_resource_patch,
    zero_action_genome,
)
from microcosmos.utils import displacement


def test_control_cppns_are_valid_and_differ_only_as_declared():
    zero = zero_action_genome()
    wave = traveling_wave_genome()
    sensory = sensory_foraging_genome()
    disabled = sensor_disabled_genome()
    for genome in (zero, wave, sensory, disabled):
        assert genome.node_genes.shape == (MAX_NODES, 5)
        assert genome.connection_genes.shape == (MAX_CONNECTIONS, 3)
        assert genome.node_genes.dtype == jnp.float32
        assert genome.connection_genes.dtype == jnp.float32
        assert bool(genome_numeric_valid(genome))

    zero_node_active = ~jnp.isnan(zero.node_genes[:, 0])
    zero_connection_active = ~jnp.isnan(zero.connection_genes[:, 0])
    assert jnp.all(zero.node_genes[zero_node_active, NODE_BIAS] == 0.0)
    assert jnp.all(zero.connection_genes[zero_connection_active, CONNECTION_WEIGHT] == 0.0)
    assert jnp.array_equal(sensory.node_genes, disabled.node_genes, equal_nan=True)
    assert jnp.array_equal(
        sensory.connection_genes[:3],
        disabled.connection_genes[:3],
        equal_nan=True,
    )
    assert jnp.any(sensory.connection_genes[3:5, CONNECTION_WEIGHT] != disabled.connection_genes[3:5, CONNECTION_WEIGHT])
    assert jnp.all(disabled.connection_genes[3:5, CONNECTION_WEIGHT] == 0.0)
    assert jnp.array_equal(
        sensory.connection_genes[5:],
        disabled.connection_genes[5:],
        equal_nan=True,
    )


def test_positive_control_environments_use_the_scientific_fluid_fixture():
    movement = movement_control_environment(horizon=10)
    foraging = foraging_control_environment(horizon=10)
    assert movement.solver_config.enable_fluid
    assert foraging.solver_config.enable_fluid
    assert movement._nodes_per_slot == foraging._nodes_per_slot == 8
    assert movement.dt == foraging.dt == 0.01
    assert foraging.resource_config.capacity == 1.0
    assert foraging.initial_resource == 0.0


def test_with_genome_recomputes_the_tensorneat_cache():
    env = movement_control_environment(horizon=10)
    _, state = env.reset(jax.random.PRNGKey(12))
    genome = zero_action_genome()
    replaced = with_genome(state, genome)
    stored = CPPNGenome(
        replaced.population.genome.node_genes[0],
        replaced.population.genome.connection_genes[0],
    )
    assert bool(
        cached_genome_valid(
            stored,
            replaced.population.controller_order[0],
            replaced.population.controller_connection_index[0],
        )
    )


def test_foraging_patch_starts_outside_the_body_and_is_map_bounded():
    env = foraging_control_environment(horizon=10)
    _, state = env.reset(jax.random.PRNGKey(0))
    patched = with_relative_resource_patch(env, state)
    body_center = env._slot_centers(state.nodes.position)[0]
    patch_center = body_center + jnp.array([-11.5, 0.0])
    node_distance = jnp.linalg.norm(displacement(env.grid_shape, state.nodes.position, patch_center), axis=-1)
    assert jnp.min(node_distance) > 4.0
    assert jnp.array_equal(patched.fields.energy, patched.resource_capacity_map)
    assert jnp.all(patched.resource_capacity_map >= 0.0)
    assert jnp.all(patched.resource_capacity_map <= env.resource_config.capacity + 1e-6)


def test_positive_control_gate_metrics_enforce_all_thresholds():
    movement = movement_gate_metrics(jnp.array([1.5, 0.0]), jnp.array([0.1, 0.0]), body_length=14.0)
    assert bool(movement["passed"])
    assert movement["body_fraction"] >= 0.1
    assert movement["drift_ratio"] >= 5.0

    foraging = foraging_gate_metrics(
        jnp.array([1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 0.9, 0.9]),
        jnp.ones(8),
    )
    assert bool(foraging["passed"])
    assert foraging["wins"] == 6
