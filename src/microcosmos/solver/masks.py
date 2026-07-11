"""Static-shape masks and topology context for fixed-capacity simulations."""

from dataclasses import dataclass
import functools

import jax
import jax.numpy as jnp
import numpy as np

from microcosmos.structs.edges import Edges
from microcosmos.structs.nodes import Nodes


ActivityMasks = tuple[jax.Array, jax.Array, jax.Array]


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=["steric_exclusion_indices", "steric_exclusion_valid"],
)
@dataclass(frozen=True)
class PhysicsContext:
    """Precomputed topology data consumed by the continuous physics kernels."""

    steric_exclusion_indices: jax.Array  # (N, K) int32
    steric_exclusion_valid: jax.Array  # (N, K) bool


def build_replicated_physics_context(
    local_pairs: jax.Array | np.ndarray,
    num_nodes: int,
    num_slots: int,
    neighbor_distance: int,
) -> PhysicsContext:
    """Build bounded graph neighborhoods and replicate them across body slots.

    The resulting lookup is independent of node storage order. Each row contains
    the node itself plus all nodes reachable within ``neighbor_distance`` edges.
    Rows are padded to a common static width for JAX kernels.
    """
    if not isinstance(num_nodes, int) or isinstance(num_nodes, bool) or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")
    if not isinstance(num_slots, int) or isinstance(num_slots, bool) or num_slots < 1:
        raise ValueError("num_slots must be a positive integer")
    if (
        not isinstance(neighbor_distance, int)
        or isinstance(neighbor_distance, bool)
        or neighbor_distance < 0
    ):
        raise ValueError("neighbor_distance must be a non-negative integer")

    pairs = np.asarray(local_pairs, dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("local_pairs must have shape (E, 2)")
    if pairs.size and (np.any(pairs < 0) or np.any(pairs >= num_nodes)):
        raise ValueError("local_pairs contains an out-of-range node index")

    adjacency = [set() for _ in range(num_nodes)]
    for source, target in pairs:
        source_i = int(source)
        target_i = int(target)
        adjacency[source_i].add(target_i)
        adjacency[target_i].add(source_i)

    neighborhoods: list[list[int]] = []
    for root in range(num_nodes):
        visited = {root}
        frontier = {root}
        for _ in range(neighbor_distance):
            frontier = {
                neighbor
                for node in frontier
                for neighbor in adjacency[node]
                if neighbor not in visited
            }
            visited.update(frontier)
            if not frontier:
                break
        neighborhoods.append(sorted(visited))

    width = max(map(len, neighborhoods))
    local_indices = np.zeros((num_nodes, width), dtype=np.int32)
    local_valid = np.zeros((num_nodes, width), dtype=np.bool_)
    for node, neighborhood in enumerate(neighborhoods):
        count = len(neighborhood)
        local_indices[node, :count] = neighborhood
        local_indices[node, count:] = node
        local_valid[node, :count] = True

    offsets = np.arange(num_slots, dtype=np.int32)[:, None, None] * num_nodes
    indices = (local_indices[None, :, :] + offsets).reshape(
        num_slots * num_nodes, width
    )
    valid = np.broadcast_to(
        local_valid[None, :, :], (num_slots, num_nodes, width)
    ).reshape(num_slots * num_nodes, width)
    return PhysicsContext(
        steric_exclusion_indices=jnp.asarray(indices),
        steric_exclusion_valid=jnp.asarray(valid),
    )


def node_activity(nodes: Nodes) -> jax.Array:
    """Return the per-node activity mask, treating legacy nodes as active."""
    if nodes.active is None:
        return jnp.ones(nodes.position.shape[0], dtype=jnp.bool_)
    return nodes.active.astype(jnp.bool_)


def edge_activity(nodes: Nodes, edges: Edges) -> jax.Array:
    """An edge is active only when both endpoint nodes are active."""
    active = node_activity(nodes)
    return active[edges.pairs[:, 0]] & active[edges.pairs[:, 1]]


def bending_activity(nodes: Nodes, edges: Edges) -> jax.Array:
    """A bending pair is active only when both participating edges are active."""
    active = edge_activity(nodes, edges)
    return active[edges.bending_pairs[:, 0]] & active[edges.bending_pairs[:, 1]]


def activity_masks(
    nodes: Nodes, edges: Edges
) -> ActivityMasks:
    """Compute node, edge, and bending masks without redundant derivation."""
    nodes_active = node_activity(nodes)
    edges_active = (
        nodes_active[edges.pairs[:, 0]] & nodes_active[edges.pairs[:, 1]]
    )
    bending_active = (
        edges_active[edges.bending_pairs[:, 0]]
        & edges_active[edges.bending_pairs[:, 1]]
    )
    return nodes_active, edges_active, bending_active
