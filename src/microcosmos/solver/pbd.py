from __future__ import annotations
from typing import TYPE_CHECKING

from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
import jax.numpy as jnp
from microcosmos.utils import displacement
from microcosmos.solver.masks import ActivityMasks, activity_masks

if TYPE_CHECKING:
    from microcosmos.solver.config import ConstraintSolverConfig


def pbd_rod_constraint(
    iter: int,
    carry: tuple[Nodes, Edges],
    config: ConstraintSolverConfig,
    grid_shape: tuple,
    masks: ActivityMasks | None = None,
) -> tuple[Nodes, Edges]:

    nodes, edges = carry
    src = edges.pairs[:, 0]  # (E,)
    tgt = edges.pairs[:, 1]  # (E,)

    # --------------------------------------------------------
    # 1. THETA PASS (Jacobi over all bending pairs)
    # --------------------------------------------------------
    bp_e_in = edges.bending_pairs[:, 0]   # (B,) edge indices
    bp_e_out = edges.bending_pairs[:, 1]  # (B,) edge indices
    masked = masks is not None or nodes.active is not None
    if masked:
        node_mask, edge_mask, bend_mask = (
            activity_masks(nodes, edges) if masks is None else masks
        )
        node_mask = node_mask.astype(nodes.position.dtype)
        edge_mask = edge_mask.astype(nodes.position.dtype)
        bend_mask = bend_mask.astype(nodes.position.dtype)

    # Edge bending degree: how many bending pairs each edge participates in
    edge_bending_degree = jnp.zeros(edges.theta.shape[0])
    degree_weight = bend_mask if masked else 1.0
    edge_bending_degree = edge_bending_degree.at[bp_e_in].add(degree_weight)
    edge_bending_degree = edge_bending_degree.at[bp_e_out].add(degree_weight)
    edge_bending_degree = jnp.maximum(edge_bending_degree, 1.0)

    theta_in = edges.theta[bp_e_in]    # (B,)
    theta_out = edges.theta[bp_e_out]  # (B,)

    # Angular difference
    diff = theta_out - theta_in  # (B,)
    diff = (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi  # wrap to [-pi, pi]

    # Correction: push toward bending rest angle
    correction = (
        (diff - edges.bending_rest_angles) * edges.bending_stiffness * 0.5
    )  # (B,)
    if masked:
        correction = correction * bend_mask

    # Scatter: e_in gets +correction, e_out gets -correction
    omega = 1.8
    delta = jnp.zeros_like(edges.theta)
    delta = delta.at[bp_e_in].add(correction)
    delta = delta.at[bp_e_out].add(-correction)
    delta = delta / edge_bending_degree * omega

    edges = edges.__replace__(theta=edges.theta + delta)

    # --------------------------------------------------------
    # 2. POSITION PASS (Jacobi over all edges)
    # --------------------------------------------------------
    # Node degree: number of edges incident on each node
    node_degree = jnp.zeros(nodes.position.shape[0])
    degree_weight = edge_mask if masked else 1.0
    node_degree = node_degree.at[src].add(degree_weight)
    node_degree = node_degree.at[tgt].add(degree_weight)
    node_degree = jnp.maximum(node_degree, 1.0)

    # Direction from per-edge theta (material frame)
    dir_x = jnp.cos(edges.theta)  # (E,)
    dir_y = jnp.sin(edges.theta)  # (E,)
    directors = jnp.stack([dir_x, dir_y], axis=-1)  # (E, 2)

    # Target vectors (material frame * rest length)
    target_vecs = directors * edges.rest_lengths[:, None]  # (E, 2)

    # Current vectors (periodic boundary aware)
    current_vecs = displacement(grid_shape, nodes.position[tgt], nodes.position[src])  # (E, 2)

    # Decompose correction into stretch and shear components
    current_lengths = jnp.linalg.norm(current_vecs, axis=-1, keepdims=True)  # (E, 1)
    current_dirs = current_vecs / jnp.maximum(current_lengths, 1e-8)  # (E, 2)

    # Stretch: length error along geometric tangent (inextensibility)
    stretch = current_dirs * (current_lengths - edges.rest_lengths[:, None])  # (E, 2)

    # Shear: directional mismatch between geometric tangent and material frame
    shear = (current_vecs - target_vecs) - stretch  # (E, 2)

    forces = (
        (stretch + shear * config.stiffness_shear)
        * config.stiffness_stretch
        * 0.5
    )  # (E, 2)
    if masked:
        forces = forces * edge_mask[:, None]

    # Scatter: src gets +forces, tgt gets -forces
    disp = jnp.zeros_like(nodes.position)
    disp = disp.at[src].add(forces)
    disp = disp.at[tgt].add(-forces)
    disp = disp / node_degree[:, None]

    if masked:
        disp = disp * node_mask[:, None]
    nodes = nodes.__replace__(position=nodes.position + disp)

    return (nodes, edges)
