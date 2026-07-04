import jax.numpy as jnp
from microcosmos.structs.fields import Fields
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.graph import make_edges, compute_bending_pairs
from microcosmos.utils import displacement


def create_dummy_nodes(num_nodes, shape=(100, 100)):
    shape = jnp.array(shape)

    # Create edge-based topology for a loop
    edge_pairs, edge_departure_angles = make_edges(
        num_nodes, graph="loop", num_instances=1
    )
    num_edges = edge_pairs.shape[0]

    # Initialize positions in a circle that matches the distance constraint
    distance_constraint = 3.0
    circle_curvature = 2.0 * jnp.pi / num_nodes
    radius = distance_constraint / (2.0 * jnp.sin(jnp.pi / num_nodes))

    angles = jnp.linspace(0, 2 * jnp.pi, num_nodes, endpoint=False)
    center = jnp.array(shape) / 2.0
    positions = center + radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    # Compute initial per-edge theta from positions
    size = tuple(shape.tolist())
    src = edge_pairs[:, 0]
    tgt = edge_pairs[:, 1]
    deltas = displacement(size, positions[tgt], positions[src])
    edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

    # Compute bending pairs
    bending_pairs, _ = compute_bending_pairs(edge_pairs, edge_departure_angles, num_nodes)

    # Bending rest angles: uniform curvature for a circle
    num_bp = bending_pairs.shape[0]
    bending_rest_angles = jnp.full(num_bp, circle_curvature)

    edge_rest_lengths = jnp.full(num_edges, distance_constraint)

    nodes = Nodes(
        position=positions,
        velocity=jnp.zeros((num_nodes, 2)),
        color_bend=jnp.arange(num_nodes) % 3,
        debug_vector=jnp.zeros((num_nodes, 2)),
    )

    edges = Edges(
        pairs=edge_pairs,
        theta=edge_theta,
        rest_lengths=edge_rest_lengths,
        bending_pairs=bending_pairs,
        bending_rest_angles=bending_rest_angles,
        bending_stiffness=jnp.ones(bending_pairs.shape[0]) * 0.5,
    )

    return nodes, edges
