from __future__ import annotations
from typing import TYPE_CHECKING

from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
import jax.numpy as jnp
from microcosmos.utils import displacement

if TYPE_CHECKING:
    from microcosmos.solver.config import ConstraintSolverConfig


def stable_cosserat_constraint(
    iter: int,
    carry: tuple[Nodes, Edges],
    config: ConstraintSolverConfig,
    grid_shape: tuple,
) -> tuple[Nodes, Edges]:
    """
    Stable Cosserat rod solver via Projective Dynamics (2D JAX port of YarnBall cosserat.cu).

    Structure mirrors YarnBall's two-kernel design:
      Phase 1 (≈ quaternionLambdaItr): update theta from bending constraints only.
      Phase 2 (≈ cosseratItr):          implicit position update with inertia term.

    The stability improvement over PBD comes entirely from Phase 2: including the
    inertia term (m/h² = 1/dt²) in the Hessian prevents oscillation at any stiffness.

    NOTE: Phase 1 does NOT couple theta to the current segment direction.
    Including such a coupling (angular residual term) caps achievable bend angles at
    π/(2*(1 + k_shear/k_b)) ≈ 0.03 rad with typical parameters, which destroys bending.
    YarnBall avoids this by using kStretch ≪ kBend for orientations (≈0.1 vs 1.0).
    For 2D the simpler and equivalent choice is a pure bending pass for theta.

    The inertial prediction y = nodes.debug_vector, set by simulate.step() before the
    fori_loop when solver_config.solver_type == "cosserat".
    """
    nodes, edges = carry

    x = nodes.position          # (N, 2) current position iterate
    y = nodes.debug_vector      # (N, 2) inertial prediction (set before loop)

    src = edges.pairs[:, 0]     # (E,)
    tgt = edges.pairs[:, 1]     # (E,)
    l = edges.rest_lengths      # (E,)

    bp_e_in  = edges.bending_pairs[:, 0]  # (B,)
    bp_e_out = edges.bending_pairs[:, 1]  # (B,)
    k_b = edges.bending_stiffness         # (B,)

    # ----------------------------------------------------------------
    # Phase 1: theta update — pure bending pass (identical to pbd.py).
    # ----------------------------------------------------------------

    edge_bending_degree = jnp.zeros(edges.theta.shape[0])
    edge_bending_degree = edge_bending_degree.at[bp_e_in].add(1.0)
    edge_bending_degree = edge_bending_degree.at[bp_e_out].add(1.0)
    edge_bending_degree = jnp.maximum(edge_bending_degree, 1.0)

    theta_in  = edges.theta[bp_e_in]
    theta_out = edges.theta[bp_e_out]
    diff = theta_out - theta_in
    diff = (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi  # wrap to [-π, π]

    correction = (diff - edges.bending_rest_angles) * k_b * 0.5

    delta = jnp.zeros_like(edges.theta)
    delta = delta.at[bp_e_in].add(correction)
    delta = delta.at[bp_e_out].add(-correction)
    delta = delta / edge_bending_degree

    theta_new = edges.theta + delta

    # Shear coupling: gently pull theta toward actual segment direction so theta
    # cannot drift arbitrarily far from geometry over multiple steps.
    # Mirrors YarnBall's quaternionLambdaItr shear term (kStretch ≪ kBend).
    seg_vec = displacement(grid_shape, x[tgt], x[src])
    alpha = jnp.arctan2(seg_vec[:, 1], seg_vec[:, 0])
    shear_pull = (alpha - theta_new + jnp.pi) % (2 * jnp.pi) - jnp.pi
    theta_new = theta_new + shear_pull * 0.05

    edges = edges.__replace__(theta=theta_new)

    # ----------------------------------------------------------------
    # Phase 2: implicit position update with inertia.
    # Mirrors YarnBall cosseratItr: per-vertex Hessian = inertia + stretch.
    #
    # Constraint: c = r/l − d(θ)  where r = x_tgt − x_src.
    # This vector encodes both length (stretch) and direction (shear).
    # ∂c/∂x_tgt =  I/l  →  H contribution: k_s/l² per endpoint (scalar·I).
    # In 2D the Hessian is exactly scalar·I (not an approximation), so
    # the per-vertex solve reduces to scalar division.
    # ----------------------------------------------------------------
    k_s = config.stiffness_stretch
    w   = 1.0 / (config.dt ** 2)   # inertia weight = m/h² (unit mass)

    d_x = jnp.cos(theta_new)       # (E,)
    d_y = jnp.sin(theta_new)       # (E,)

    r = displacement(grid_shape, x[tgt], x[src])               # (E, 2)
    c = r / l[:, None] - jnp.stack([d_x, d_y], axis=-1)       # (E, 2)

    f_edge = (k_s / l)[:, None] * c    # (E, 2) — restoring force per edge
    H_edge = k_s / (l ** 2)            # (E,)  — Hessian contribution per edge

    # Inertia: pulls position toward the inertial prediction y
    f_node = w * (y - x)               # (N, 2)
    H_node = jnp.full(x.shape[0], w)  # (N,)

    # Scatter: r = x_tgt − x_src, so restoring force is +c on src, −c on tgt
    f_node = f_node.at[src].add( f_edge)
    f_node = f_node.at[tgt].add(-f_edge)
    H_node = H_node.at[src].add(H_edge)
    H_node = H_node.at[tgt].add(H_edge)

    x_new = x + f_node / H_node[:, None]
    nodes = nodes.__replace__(position=x_new)

    return nodes, edges
