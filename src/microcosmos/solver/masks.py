"""Static-shape activity masks for fixed-capacity simulations."""

import jax
import jax.numpy as jnp

from microcosmos.structs.edges import Edges
from microcosmos.structs.nodes import Nodes


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
) -> tuple[jax.Array, jax.Array, jax.Array]:
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
