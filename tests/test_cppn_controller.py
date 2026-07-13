import hashlib
import math

import jax
import jax.numpy as jnp
import pytest

from microcosmos.cppn import (
    CONNECTION_INPUT,
    CONNECTION_OUTPUT,
    CONNECTION_WEIGHT,
    HIDDEN_ROW,
    IDENTITY_ACTIVATION,
    MAX_CONNECTIONS,
    MAX_NODES,
    NODE_ACTIVATION,
    NODE_AGGREGATION,
    NODE_BIAS,
    NODE_KEY,
    NODE_RESPONSE,
    NUM_INPUTS,
    OUTPUT_ROW,
    SINE_ACTIVATION,
    SUM_AGGREGATION,
    CPPNGenome,
    cached_genome_valid,
    canonical_cppn_genome,
    controller_action,
    genome_numeric_valid,
    initialize_cppn_population,
    transform_and_validate_genome,
    transform_population,
)


def _equal_with_nan(left: jax.Array, right: jax.Array) -> bool:
    return bool(jnp.all((jnp.isnan(left) & jnp.isnan(right)) | (left == right)))


def test_canonical_cppn_layout_is_exact_and_valid():
    genome = canonical_cppn_genome()

    assert genome.node_genes.shape == (MAX_NODES, 5)
    assert genome.connection_genes.shape == (MAX_CONNECTIONS, 3)
    assert genome.node_genes.dtype == jnp.float32
    assert genome.connection_genes.dtype == jnp.float32
    assert jnp.array_equal(
        genome.node_genes[:6, NODE_KEY],
        jnp.arange(6, dtype=jnp.float32),
    )
    assert jnp.all(jnp.isnan(genome.node_genes[6:]))
    assert jnp.all(genome.node_genes[:6, NODE_BIAS] == 0.0)
    assert jnp.all(genome.node_genes[:6, NODE_RESPONSE] == 1.0)
    assert jnp.all(genome.node_genes[:6, NODE_AGGREGATION] == SUM_AGGREGATION)
    assert jnp.all(genome.node_genes[:NUM_INPUTS, NODE_ACTIVATION] == IDENTITY_ACTIVATION)
    assert genome.node_genes[HIDDEN_ROW, NODE_ACTIVATION] == SINE_ACTIVATION
    assert genome.node_genes[OUTPUT_ROW, NODE_ACTIVATION] == IDENTITY_ACTIVATION

    expected_connections = jnp.array(
        [
            [0.0, 4.0, math.pi],
            [1.0, 4.0, -0.80],
            [4.0, 5.0, 1.00],
            [2.0, 5.0, 1.00],
            [3.0, 5.0, 0.50],
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(genome.connection_genes[:5], expected_connections)
    assert jnp.all(jnp.isnan(genome.connection_genes[5:]))

    canonical, order, connection_index, valid = transform_and_validate_genome(genome)
    assert valid
    assert cached_genome_valid(canonical, order, connection_index)


def test_founder_panel_is_deterministic_and_only_living_slots_are_varied():
    first = initialize_cppn_population(capacity=5, initial_population=3)
    second = initialize_cppn_population(capacity=5, initial_population=3)
    canonical = canonical_cppn_genome()

    assert _equal_with_nan(first.node_genes, second.node_genes)
    assert _equal_with_nan(first.connection_genes, second.connection_genes)
    for slot in range(3, 5):
        assert _equal_with_nan(first.node_genes[slot], canonical.node_genes)
        assert _equal_with_nan(first.connection_genes[slot], canonical.connection_genes)
    assert any(not _equal_with_nan(first.connection_genes[slot], canonical.connection_genes) for slot in range(3))


def test_explicit_panel_key_preserves_default_bytes_and_selects_new_panel():
    default = initialize_cppn_population(capacity=5, initial_population=3)
    explicit_default = initialize_cppn_population(
        capacity=5,
        initial_population=3,
        panel_key=jax.random.PRNGKey(0),
    )
    fresh = initialize_cppn_population(
        capacity=5,
        initial_population=3,
        panel_key=jax.random.PRNGKey(5_005),
    )

    assert jax.device_get(default.node_genes).tobytes() == jax.device_get(
        explicit_default.node_genes
    ).tobytes()
    assert jax.device_get(default.connection_genes).tobytes() == jax.device_get(
        explicit_default.connection_genes
    ).tobytes()
    assert not _equal_with_nan(default.node_genes[:3], fresh.node_genes[:3])
    assert not _equal_with_nan(
        default.connection_genes[:3], fresh.connection_genes[:3]
    )
    assert _equal_with_nan(default.node_genes[3:], fresh.node_genes[3:])
    assert _equal_with_nan(
        default.connection_genes[3:], fresh.connection_genes[3:]
    )


def test_explicit_founder_and_panel_key_are_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot be combined"):
        initialize_cppn_population(
            capacity=2,
            initial_population=1,
            founder_genome=canonical_cppn_genome(),
            panel_key=jax.random.PRNGKey(5_005),
        )


def test_r5_panel_seed_repeats_at_the_frozen_full_capacity_digest():
    def panel_bytes():
        panel = initialize_cppn_population(
            capacity=80,
            initial_population=80,
            panel_key=jax.random.PRNGKey(5_005),
        )
        return (
            jax.device_get(panel.node_genes).tobytes()
            + jax.device_get(panel.connection_genes).tobytes()
        )

    first = panel_bytes()
    second = panel_bytes()
    assert first == second
    assert hashlib.sha256(first).hexdigest() == (
        "9b3c479771bd8f020e0bc072827af28d535d26610c089b06a33f4d221e85628f"
    )


def test_explicit_founder_is_canonicalized_and_broadcast_across_every_slot():
    canonical = canonical_cppn_genome()
    founder = CPPNGenome(
        node_genes=canonical.node_genes.at[HIDDEN_ROW, NODE_BIAS].set(0.75).at[0, NODE_BIAS].set(3.0),
        connection_genes=canonical.connection_genes.at[0, CONNECTION_WEIGHT].set(2.5),
    )

    population = initialize_cppn_population(
        capacity=5,
        initial_population=2,
        founder_genome=founder,
    )

    assert population.node_genes.shape == (5, MAX_NODES, 5)
    assert population.connection_genes.shape == (5, MAX_CONNECTIONS, 3)
    for slot in range(5):
        assert _equal_with_nan(population.node_genes[slot], population.node_genes[0])
        assert _equal_with_nan(
            population.connection_genes[slot],
            population.connection_genes[0],
        )
    assert population.node_genes[0, HIDDEN_ROW, NODE_BIAS] == 0.75
    assert population.connection_genes[0, 0, CONNECTION_WEIGHT] == 2.5
    assert population.node_genes[0, 0, NODE_BIAS] == 0.0


def test_explicit_founder_rejects_invalid_genome():
    canonical = canonical_cppn_genome()
    invalid = CPPNGenome(
        canonical.node_genes,
        canonical.connection_genes.at[0, CONNECTION_OUTPUT].set(99.0),
    )

    with pytest.raises(ValueError, match="founder_genome"):
        initialize_cppn_population(
            capacity=3,
            initial_population=2,
            founder_genome=invalid,
        )


@pytest.mark.parametrize(
    ("capacity", "initial_population"),
    [(0, 0), (True, 0), (2, -1), (2, 3), (2, True)],
)
def test_founder_panel_rejects_invalid_capacity_or_population(capacity, initial_population):
    with pytest.raises(ValueError):
        initialize_cppn_population(capacity=capacity, initial_population=initial_population)


def test_interleaved_all_nan_holes_are_legal_but_partial_nan_rows_are_not():
    genome = canonical_cppn_genome()
    nodes = genome.node_genes.at[10].set(genome.node_genes[HIDDEN_ROW]).at[HIDDEN_ROW].set(jnp.nan)
    connections = genome.connection_genes.at[10].set(genome.connection_genes[0]).at[0].set(jnp.nan)
    interleaved = CPPNGenome(nodes, connections)

    assert genome_numeric_valid(interleaved)
    assert transform_and_validate_genome(interleaved)[-1]

    partial_node = CPPNGenome(nodes.at[7, NODE_KEY].set(12.0), connections)
    partial_connection = CPPNGenome(nodes, connections.at[7, CONNECTION_INPUT].set(0.0))
    assert not genome_numeric_valid(partial_node)
    assert not genome_numeric_valid(partial_connection)


def test_positional_inputs_and_output_are_mandatory():
    genome = canonical_cppn_genome()
    swapped_inputs = CPPNGenome(
        genome.node_genes.at[0].set(genome.node_genes[1]).at[1].set(genome.node_genes[0]),
        genome.connection_genes,
    )
    wrong_output = CPPNGenome(
        genome.node_genes.at[OUTPUT_ROW, NODE_KEY].set(12.0),
        genome.connection_genes,
    )

    assert not transform_and_validate_genome(swapped_inputs)[-1]
    assert not transform_and_validate_genome(wrong_output)[-1]


def test_cached_transform_detects_stale_topology_but_accepts_current_cache():
    genome = canonical_cppn_genome()
    _, old_order, old_connection_index, valid = transform_and_validate_genome(genome)
    assert valid

    moved_connections = genome.connection_genes.at[10].set(genome.connection_genes[0]).at[0].set(jnp.nan)
    moved = CPPNGenome(genome.node_genes, moved_connections)
    canonical, order, connection_index, valid = transform_and_validate_genome(moved)

    assert valid
    assert cached_genome_valid(canonical, order, connection_index)
    assert not cached_genome_valid(canonical, old_order, old_connection_index)


def test_controller_is_eager_jit_equivalent_finite_bounded_and_alive_masked():
    genomes = initialize_cppn_population(capacity=3, initial_population=3)
    order, connection_index = transform_population(genomes)
    coordinates = jnp.tile(jnp.array([-0.8, 0.0, 0.8], dtype=jnp.float32), 3)
    local_resource = jnp.array(
        [0.0, 0.5, 1.0, -jnp.inf, jnp.inf, 0.3, 0.9, 0.2, 0.6],
        dtype=jnp.float32,
    )
    head_resource = jnp.array([0.9, jnp.inf, 0.1], dtype=jnp.float32)
    tail_resource = jnp.array([0.1, -jnp.inf, 0.8], dtype=jnp.float32)
    alive = jnp.array([True, False, True])
    bound = 0.27

    def act():
        return controller_action(
            genomes,
            order,
            connection_index,
            coordinates,
            jnp.array(0.75, dtype=jnp.float32),
            local_resource,
            head_resource,
            tail_resource,
            alive,
            bound,
        )

    eager = act()
    compiled = jax.jit(act)()
    assert eager.shape == coordinates.shape
    assert jnp.all(jnp.isfinite(eager))
    assert jnp.all(jnp.abs(eager) <= bound)
    assert jnp.all(eager[3:6] == 0.0)
    assert jnp.allclose(eager, compiled, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("defect", ["duplicate_node", "dangling_endpoint", "duplicate_connection", "cycle"])
def test_full_validation_rejects_invalid_graphs(defect):
    genome = canonical_cppn_genome()
    nodes = genome.node_genes
    connections = genome.connection_genes
    if defect == "duplicate_node":
        nodes = nodes.at[6].set(nodes[HIDDEN_ROW])
    elif defect == "dangling_endpoint":
        connections = connections.at[0, CONNECTION_OUTPUT].set(99.0)
    elif defect == "duplicate_connection":
        connections = connections.at[5].set(connections[0])
    else:
        connections = connections.at[5].set(jnp.array([5.0, 4.0, 1.0], dtype=jnp.float32))

    assert not transform_and_validate_genome(CPPNGenome(nodes, connections))[-1]


def test_full_validation_rejects_out_of_bounds_attributes():
    genome = canonical_cppn_genome()
    bad_node = CPPNGenome(genome.node_genes.at[HIDDEN_ROW, NODE_BIAS].set(6.0), genome.connection_genes)
    bad_connection = CPPNGenome(
        genome.node_genes,
        genome.connection_genes.at[0, CONNECTION_WEIGHT].set(6.0),
    )
    assert not transform_and_validate_genome(bad_node)[-1]
    assert not transform_and_validate_genome(bad_connection)[-1]
