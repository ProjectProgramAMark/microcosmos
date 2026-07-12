"""Fixed-capacity CPPN controllers for the evolving ecosystem."""

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tensorneat as neat
from tensorneat.common import ACT, AGG, I_INF, State


NUM_INPUTS = 4
NUM_OUTPUTS = 1
MAX_NODES = 15
MAX_CONNECTIONS = 30

INPUT_ROWS = (0, 1, 2, 3)
HIDDEN_ROW = 4
OUTPUT_ROW = 5
DYNAMIC_NODE_KEY_OFFSET = 1024
MAX_EXACT_FLOAT32_INTEGER = 2**24
FOUNDER_PANEL_SEED = 0
PHASE_RATE = 5.0
FOUNDER_VARIATION_STD = 0.03
RAW_OUTPUT_LIMIT = 20.0

NODE_KEY = 0
NODE_BIAS = 1
NODE_RESPONSE = 2
NODE_AGGREGATION = 3
NODE_ACTIVATION = 4

CONNECTION_INPUT = 0
CONNECTION_OUTPUT = 1
CONNECTION_WEIGHT = 2

IDENTITY_ACTIVATION = 0
SINE_ACTIVATION = 1
TANH_ACTIVATION = 2
ABS_ACTIVATION = 3
SUM_AGGREGATION = 0

WEIGHT_LOWER_BOUND = -5.0
WEIGHT_UPPER_BOUND = 5.0
NODE_ATTRIBUTE_LOWER_BOUND = -5.0
NODE_ATTRIBUTE_UPPER_BOUND = 5.0


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=["node_genes", "connection_genes"],
)
@dataclass
class CPPNGenome:
    """TensorNEAT node and connection genes, optionally batched by organism."""

    node_genes: jax.Array
    connection_genes: jax.Array


def build_tensorneat_genome(
    *,
    value_mutation: bool,
    structural_mutation: bool,
) -> neat.genome.DefaultGenome:
    """Build a shape-compatible TensorNEAT genome with frozen mutation rates."""
    value_rate = 0.20 if value_mutation else 0.0
    value_power = 0.15 if value_mutation else 0.0
    replace_rate = 0.015 if value_mutation else 0.0
    activation_rate = 0.10 if value_mutation else 0.0
    conn_rate = 0.20 if structural_mutation else 0.0
    node_rate = 0.10 if structural_mutation else 0.0

    return neat.genome.DefaultGenome(
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNECTIONS,
        init_hidden_layers=(1,),
        node_gene=neat.genome.DefaultNode(
            bias_mutate_power=value_power,
            bias_mutate_rate=value_rate,
            bias_replace_rate=replace_rate,
            bias_lower_bound=NODE_ATTRIBUTE_LOWER_BOUND,
            bias_upper_bound=NODE_ATTRIBUTE_UPPER_BOUND,
            response_mutate_power=value_power,
            response_mutate_rate=value_rate,
            response_replace_rate=replace_rate,
            response_lower_bound=NODE_ATTRIBUTE_LOWER_BOUND,
            response_upper_bound=NODE_ATTRIBUTE_UPPER_BOUND,
            aggregation_options=[AGG.sum],
            aggregation_default=AGG.sum,
            aggregation_replace_rate=0.0,
            activation_options=[ACT.identity, ACT.sin, ACT.tanh, ACT.abs],
            activation_default=ACT.identity,
            activation_replace_rate=activation_rate,
        ),
        conn_gene=neat.genome.DefaultConn(
            weight_mutate_power=value_power,
            weight_mutate_rate=value_rate,
            weight_replace_rate=replace_rate,
            weight_lower_bound=WEIGHT_LOWER_BOUND,
            weight_upper_bound=WEIGHT_UPPER_BOUND,
        ),
        mutation=neat.genome.DefaultMutation(
            conn_add=conn_rate,
            conn_delete=conn_rate,
            node_add=node_rate,
            node_delete=node_rate,
        ),
        output_transform=ACT.identity,
    )


INFERENCE_GENOME = build_tensorneat_genome(
    value_mutation=False,
    structural_mutation=False,
)
INFERENCE_STATE = INFERENCE_GENOME.setup(State())


def canonical_cppn_genome() -> CPPNGenome:
    """Return a valid traveling-wave CPPN with small resource corrections."""
    nodes = jnp.full((MAX_NODES, 5), jnp.nan, dtype=jnp.float32)
    base_rows = jnp.stack(
        [
            jnp.arange(6, dtype=jnp.float32),
            jnp.zeros(6, dtype=jnp.float32),
            jnp.ones(6, dtype=jnp.float32),
            jnp.zeros(6, dtype=jnp.float32),
            jnp.zeros(6, dtype=jnp.float32),
        ],
        axis=-1,
    )
    nodes = nodes.at[:6].set(base_rows)
    nodes = nodes.at[HIDDEN_ROW, NODE_ACTIVATION].set(SINE_ACTIVATION)

    connections = jnp.full((MAX_CONNECTIONS, 3), jnp.nan, dtype=jnp.float32)
    founder_connections = jnp.array(
        [
            [0.0, 4.0, math.pi],
            [1.0, 4.0, -0.80],
            [4.0, 5.0, 1.00],
            [2.0, 5.0, 1.00],
            [3.0, 5.0, 0.50],
        ],
        dtype=jnp.float32,
    )
    connections = connections.at[: founder_connections.shape[0]].set(founder_connections)
    return CPPNGenome(nodes, connections)


def canonicalize_neutral_attributes(genome: CPPNGenome) -> CPPNGenome:
    """Reset TensorNEAT attributes that the feed-forward ABI does not consume."""
    nodes = genome.node_genes
    input_identity = jnp.array(
        [0.0, 1.0, SUM_AGGREGATION, IDENTITY_ACTIVATION],
        dtype=nodes.dtype,
    )
    nodes = nodes.at[:NUM_INPUTS, 1:].set(input_identity)
    nodes = nodes.at[OUTPUT_ROW, NODE_ACTIVATION].set(IDENTITY_ACTIVATION)
    return CPPNGenome(nodes, genome.connection_genes)


def initialize_cppn_population(
    capacity: int,
    initial_population: int,
) -> CPPNGenome:
    """Create a canonical fixed-capacity population with varied living founders."""
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    if not isinstance(initial_population, int) or isinstance(initial_population, bool) or not 0 <= initial_population <= capacity:
        raise ValueError("initial_population must be within capacity")

    canonical = canonical_cppn_genome()
    nodes = jnp.broadcast_to(canonical.node_genes, (capacity, *canonical.node_genes.shape))
    connections = jnp.broadcast_to(
        canonical.connection_genes,
        (capacity, *canonical.connection_genes.shape),
    )
    founder_key = jax.random.PRNGKey(FOUNDER_PANEL_SEED)
    node_key, connection_key = jax.random.split(founder_key)

    node_noise = jax.random.normal(node_key, (capacity, 2, 2), dtype=jnp.float32) * FOUNDER_VARIATION_STD
    varied_nodes = nodes.at[:, HIDDEN_ROW : OUTPUT_ROW + 1, NODE_BIAS].add(node_noise[..., 0])
    varied_nodes = varied_nodes.at[:, HIDDEN_ROW : OUTPUT_ROW + 1, NODE_RESPONSE].add(node_noise[..., 1])
    varied_nodes = varied_nodes.at[..., NODE_BIAS].set(
        jnp.clip(
            varied_nodes[..., NODE_BIAS],
            NODE_ATTRIBUTE_LOWER_BOUND,
            NODE_ATTRIBUTE_UPPER_BOUND,
        )
    )
    varied_nodes = varied_nodes.at[..., NODE_RESPONSE].set(
        jnp.clip(
            varied_nodes[..., NODE_RESPONSE],
            NODE_ATTRIBUTE_LOWER_BOUND,
            NODE_ATTRIBUTE_UPPER_BOUND,
        )
    )

    connection_noise = (
        jax.random.normal(
            connection_key,
            (capacity, 5),
            dtype=jnp.float32,
        )
        * FOUNDER_VARIATION_STD
    )
    varied_connections = connections.at[:, :5, CONNECTION_WEIGHT].add(connection_noise)
    varied_connections = varied_connections.at[..., CONNECTION_WEIGHT].set(
        jnp.clip(
            varied_connections[..., CONNECTION_WEIGHT],
            WEIGHT_LOWER_BOUND,
            WEIGHT_UPPER_BOUND,
        )
    )

    living = jnp.arange(capacity) < initial_population
    nodes = jnp.where(living[:, None, None], varied_nodes, nodes)
    connections = jnp.where(living[:, None, None], varied_connections, connections)
    return jax.vmap(canonicalize_neutral_attributes)(CPPNGenome(nodes, connections))


def transform_genome(genome: CPPNGenome) -> tuple[jax.Array, jax.Array]:
    """Compute the two derived TensorNEAT arrays cached by the ecosystem."""
    order, _, _, connection_index = INFERENCE_GENOME.transform(
        INFERENCE_STATE,
        genome.node_genes,
        genome.connection_genes,
    )
    return order, connection_index


def transform_population(genome: CPPNGenome) -> tuple[jax.Array, jax.Array]:
    """Transform a fixed-capacity batch of CPPN genomes."""
    return jax.vmap(transform_genome)(genome)


def _forward_organism(
    node_genes: jax.Array,
    connection_genes: jax.Array,
    order: jax.Array,
    connection_index: jax.Array,
    observations: jax.Array,
) -> jax.Array:
    transformed = (order, node_genes, connection_genes, connection_index)
    return jax.vmap(lambda values: INFERENCE_GENOME.forward(INFERENCE_STATE, transformed, values)[0])(observations)


def controller_action(
    genome: CPPNGenome,
    controller_order: jax.Array,
    controller_connection_index: jax.Array,
    normalized_coordinate: jax.Array,
    physical_time: jax.Array,
    local_resource: jax.Array,
    head_resource: jax.Array,
    tail_resource: jax.Array,
    alive: jax.Array,
    max_bending_delta: float,
) -> jax.Array:
    """Run all organism CPPNs over every bending hinge."""
    capacity = alive.shape[0]
    if normalized_coordinate.shape[0] == 0:
        return jnp.zeros_like(normalized_coordinate)
    if normalized_coordinate.shape[0] % capacity:
        raise ValueError("bending coordinates must divide evenly by capacity")

    hinges_per_organism = normalized_coordinate.shape[0] // capacity
    coordinate = normalized_coordinate.reshape(capacity, hinges_per_organism)
    local = jnp.clip(local_resource, 0.0, 1.0).reshape(capacity, hinges_per_organism)
    phase = jnp.broadcast_to(
        jnp.asarray(physical_time, dtype=jnp.float32) * PHASE_RATE,
        coordinate.shape,
    )
    gradient = jnp.broadcast_to(
        jnp.clip(head_resource - tail_resource, -1.0, 1.0)[:, None],
        coordinate.shape,
    )
    observations = jnp.stack([coordinate, phase, local, gradient], axis=-1)

    raw = jax.vmap(_forward_organism)(
        genome.node_genes,
        genome.connection_genes,
        controller_order,
        controller_connection_index,
        observations,
    )
    raw = jnp.nan_to_num(
        raw,
        nan=0.0,
        posinf=RAW_OUTPUT_LIMIT,
        neginf=-RAW_OUTPUT_LIMIT,
    )
    action = max_bending_delta * jnp.tanh(raw)
    return (action * alive[:, None]).reshape(-1)


def _rows_numeric_valid(rows: jax.Array) -> jax.Array:
    all_nan = jnp.all(jnp.isnan(rows), axis=-1)
    all_finite = jnp.all(jnp.isfinite(rows), axis=-1)
    return jnp.all(all_nan | all_finite)


def genome_numeric_valid(genome: CPPNGenome) -> jax.Array:
    """Check legal complete rows versus all-NaN unused rows."""
    return _rows_numeric_valid(genome.node_genes) & _rows_numeric_valid(genome.connection_genes)


def population_numeric_valid(genome: CPPNGenome) -> jax.Array:
    """Check legal numeric structure for a batched population."""
    return jnp.all(jax.vmap(genome_numeric_valid)(genome))


def _no_duplicate_nodes(nodes: jax.Array) -> jax.Array:
    active = ~jnp.isnan(nodes[:, NODE_KEY])
    same = nodes[:, None, NODE_KEY] == nodes[None, :, NODE_KEY]
    duplicated = same & active[:, None] & active[None, :]
    duplicated = duplicated & ~jnp.eye(MAX_NODES, dtype=jnp.bool_)
    return ~jnp.any(duplicated)


def _connections_valid(
    nodes: jax.Array,
    connections: jax.Array,
) -> jax.Array:
    node_active = ~jnp.isnan(nodes[:, NODE_KEY])
    connection_active = ~jnp.isnan(connections[:, CONNECTION_INPUT])
    keys = nodes[:, NODE_KEY]

    input_exists = jnp.any(
        connections[:, CONNECTION_INPUT, None] == keys[None, :],
        axis=-1,
    )
    output_exists = jnp.any(
        connections[:, CONNECTION_OUTPUT, None] == keys[None, :],
        axis=-1,
    )
    endpoints_valid = jnp.all(~connection_active | (input_exists & output_exists))

    same_input = connections[:, None, CONNECTION_INPUT] == connections[None, :, CONNECTION_INPUT]
    same_output = connections[:, None, CONNECTION_OUTPUT] == connections[None, :, CONNECTION_OUTPUT]
    duplicated = same_input & same_output & connection_active[:, None] & connection_active[None, :] & ~jnp.eye(MAX_CONNECTIONS, dtype=jnp.bool_)
    integer_endpoints = jnp.all(
        ~connection_active
        | (
            (connections[:, CONNECTION_INPUT] == jnp.round(connections[:, CONNECTION_INPUT]))
            & (connections[:, CONNECTION_OUTPUT] == jnp.round(connections[:, CONNECTION_OUTPUT]))
        )
    )
    return jnp.any(node_active) & endpoints_valid & ~jnp.any(duplicated) & integer_endpoints


def transform_and_validate_genome(
    genome: CPPNGenome,
) -> tuple[CPPNGenome, jax.Array, jax.Array, jax.Array]:
    """Canonicalize, transform, and fully validate one newborn controller."""
    if genome.node_genes.shape != (MAX_NODES, 5):
        raise ValueError("node_genes must have shape (15, 5)")
    if genome.connection_genes.shape != (MAX_CONNECTIONS, 3):
        raise ValueError("connection_genes must have shape (30, 3)")
    if genome.node_genes.dtype != jnp.float32:
        raise TypeError("node_genes must have dtype float32")
    if genome.connection_genes.dtype != jnp.float32:
        raise TypeError("connection_genes must have dtype float32")
    genome = canonicalize_neutral_attributes(genome)
    nodes = genome.node_genes
    connections = genome.connection_genes
    order, connection_index = transform_genome(genome)

    node_active = ~jnp.isnan(nodes[:, NODE_KEY])
    connection_active = ~jnp.isnan(connections[:, CONNECTION_INPUT])
    positional_io = jnp.all(nodes[:NUM_INPUTS, NODE_KEY] == jnp.arange(NUM_INPUTS)) & (nodes[OUTPUT_ROW, NODE_KEY] == OUTPUT_ROW)
    integer_node_fields = jnp.all(
        ~node_active
        | (
            (nodes[:, NODE_KEY] == jnp.round(nodes[:, NODE_KEY]))
            & (nodes[:, NODE_AGGREGATION] == jnp.round(nodes[:, NODE_AGGREGATION]))
            & (nodes[:, NODE_ACTIVATION] == jnp.round(nodes[:, NODE_ACTIVATION]))
        )
    )
    node_key_bounds = jnp.all(~node_active | ((nodes[:, NODE_KEY] >= 0) & (nodes[:, NODE_KEY] < MAX_EXACT_FLOAT32_INTEGER)))
    node_categories = jnp.all(
        ~node_active
        | ((nodes[:, NODE_AGGREGATION] == SUM_AGGREGATION) & (nodes[:, NODE_ACTIVATION] >= IDENTITY_ACTIVATION) & (nodes[:, NODE_ACTIVATION] <= ABS_ACTIVATION))
    )
    node_bounds = jnp.all(
        ~node_active
        | (
            (nodes[:, NODE_BIAS] >= NODE_ATTRIBUTE_LOWER_BOUND)
            & (nodes[:, NODE_BIAS] <= NODE_ATTRIBUTE_UPPER_BOUND)
            & (nodes[:, NODE_RESPONSE] >= NODE_ATTRIBUTE_LOWER_BOUND)
            & (nodes[:, NODE_RESPONSE] <= NODE_ATTRIBUTE_UPPER_BOUND)
        )
    )
    connection_bounds = jnp.all(
        ~connection_active | ((connections[:, CONNECTION_WEIGHT] >= WEIGHT_LOWER_BOUND) & (connections[:, CONNECTION_WEIGHT] <= WEIGHT_UPPER_BOUND))
    )
    order_valid = jnp.all((order == I_INF) | ((order >= 0) & (order < MAX_NODES)))
    connection_index_valid = jnp.all((connection_index == I_INF) | ((connection_index >= 0) & (connection_index < MAX_CONNECTIONS)))
    acyclic = jnp.sum(order != I_INF) == jnp.sum(node_active)
    valid = (
        genome_numeric_valid(genome)
        & positional_io
        & integer_node_fields
        & node_key_bounds
        & node_categories
        & node_bounds
        & connection_bounds
        & _no_duplicate_nodes(nodes)
        & _connections_valid(nodes, connections)
        & order_valid
        & connection_index_valid
        & acyclic
    )
    return genome, order, connection_index, valid


def transform_and_validate_population(
    genome: CPPNGenome,
) -> tuple[CPPNGenome, jax.Array, jax.Array, jax.Array]:
    """Canonicalize, transform, and validate a fixed-capacity population."""
    return jax.vmap(transform_and_validate_genome)(genome)


def cached_genome_valid(
    genome: CPPNGenome,
    order: jax.Array,
    connection_index: jax.Array,
) -> jax.Array:
    """Validate a genome and prove its cached transform is current."""
    canonical, expected_order, expected_index, valid = transform_and_validate_genome(genome)
    same_nodes = jnp.all((jnp.isnan(canonical.node_genes) & jnp.isnan(genome.node_genes)) | (canonical.node_genes == genome.node_genes))
    same_connections = jnp.all(
        (jnp.isnan(canonical.connection_genes) & jnp.isnan(genome.connection_genes)) | (canonical.connection_genes == genome.connection_genes)
    )
    return valid & same_nodes & same_connections & jnp.array_equal(order, expected_order) & jnp.array_equal(connection_index, expected_index)
