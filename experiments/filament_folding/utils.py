import jax
import jax.numpy as jnp

from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.graph import make_edges, initialize_positions, compute_bending_pairs
from microcosmos.utils import displacement


def init_nodes(num_nodes, shape, initialization="loop", init_line_distance=2.0) -> tuple[Nodes, Edges]:
    key = jax.random.PRNGKey(0)
    shape = jnp.array(shape)
    num_instances = 1

    edge_pairs, edge_departure_angles = make_edges(
        num_nodes, graph=initialization, num_instances=num_instances
    )
    positions = initialize_positions(num_nodes, initialization, shape, key, num_instances=num_instances)
    velocity = jnp.zeros_like(positions)
    size = tuple(shape.tolist())

    num_edges = edge_pairs.shape[0]
    src = edge_pairs[:, 0]
    tgt = edge_pairs[:, 1]
    deltas = displacement(size, positions[tgt], positions[src])
    edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

    bending_pairs, bending_rest_angles = compute_bending_pairs(edge_pairs, edge_departure_angles, num_nodes)
    edge_rest_lengths = jnp.full(num_edges, init_line_distance)

    nodes = Nodes(
        position=positions,
        velocity=velocity,
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


def downsample_particles_to_grid(nodes: Nodes, target_size: int, sdf_cap: int, grid_shape: tuple):
    H, W = grid_shape
    scale_x = W / target_size
    scale_y = H / target_size

    y_coords = jnp.arange(target_size)[:, None] * scale_y + scale_y / 2
    x_coords = jnp.arange(target_size)[None, :] * scale_x + scale_x / 2

    positions = nodes.position  # (N, 2)

    dx = x_coords[:, :, None] - positions[None, None, :, 0]
    dy = y_coords[:, :, None] - positions[None, None, :, 1]
    dx = jnp.minimum(jnp.abs(dx), W - jnp.abs(dx))
    dy = jnp.minimum(jnp.abs(dy), H - jnp.abs(dy))
    dist = jnp.sqrt(dx ** 2 + dy ** 2)
    smallest_distance = jnp.min(dist, axis=-1)
    scale = target_size / max(H, W)
    smallest_distance = smallest_distance * scale
    smallest_distance = jnp.minimum(smallest_distance, sdf_cap)
    normalized_distance = 1 - smallest_distance / (jnp.max(smallest_distance) + 1e-8)
    return normalized_distance
