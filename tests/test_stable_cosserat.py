"""
Minimal tests for the stable Cosserat solver.

Three invariants that must hold regardless of implementation details:
1. Segment lengths are maintained (stretch constraint).
2. Bending angles converge toward rest angles (bending constraint).
3. A straight chain with zero rest angles stays straight (trivial equilibrium).
"""

import jax.numpy as jnp
import pytest
from microcosmos.solver.config import COSSERAT_SCHEME_NO_FLUID
from microcosmos.solver.stable_cosserat import stable_cosserat_constraint
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.simulate import step
from microcosmos.structs.fields import Fields
from microcosmos.utils import displacement


GRID = (64, 64)


def _make_line(n, rest_length=1.0):
    """Horizontal line of n nodes, all thetas=0, zero bending rest angles."""
    from microcosmos.graph import make_edges, compute_bending_pairs
    pos = jnp.stack([jnp.arange(n, dtype=float) * rest_length,
                     jnp.full(n, 32.0)], axis=-1)
    pairs, dep_angles = make_edges(n, graph="line", num_instances=1)
    bp, bra = compute_bending_pairs(pairs, dep_angles, n)
    nodes = Nodes(
        position=pos,
        velocity=jnp.zeros_like(pos),
        color_bend=jnp.zeros(n),
        debug_vector=jnp.zeros_like(pos),
    )
    edges = Edges(
        pairs=pairs,
        theta=jnp.zeros(pairs.shape[0]),
        rest_lengths=jnp.full(pairs.shape[0], rest_length),
        bending_pairs=bp,
        bending_rest_angles=bra,
        bending_stiffness=jnp.ones(bp.shape[0]),
    )
    return nodes, edges


def _fields():
    return Fields(
        grid_shape=GRID,
        steric=jnp.zeros(GRID),
        fluid_velocity=jnp.zeros((2, *GRID)),
        f_grid=jnp.zeros((9, *GRID)),
    )


def _seg_lengths(nodes, edges):
    r = displacement(GRID, nodes.position[edges.pairs[:, 1]],
                            nodes.position[edges.pairs[:, 0]])
    return jnp.linalg.norm(r, axis=-1)


def _bend_diffs(nodes, edges):
    """Current theta differences across bending pairs (unwrapped)."""
    bp = edges.bending_pairs
    diff = edges.theta[bp[:, 1]] - edges.theta[bp[:, 0]]
    return (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _run_steps(nodes, edges, n=30):
    fields = _fields()
    for _ in range(n):
        nodes, edges, fields = step(nodes, edges, fields, 0.05, COSSERAT_SCHEME_NO_FLUID)
    return nodes, edges


# ──────────────────────────────────────────────────────────
# 1. Straight equilibrium
# ──────────────────────────────────────────────────────────

def test_straight_line_stable():
    """Straight line at rest length with zero rest angles should not move."""
    n = 10
    nodes0, edges0 = _make_line(n)
    nodes1, edges1 = _run_steps(nodes0, edges0, n=20)

    pos_shift = jnp.abs(nodes1.position - nodes0.position).max()
    assert pos_shift < 0.01, f"straight line drifted: {pos_shift:.4f}"


# ──────────────────────────────────────────────────────────
# 2. Stretch constraint
# ──────────────────────────────────────────────────────────

def test_stretch_maintains_length():
    """Segment lengths must stay close to rest_length after many steps."""
    nodes, edges = _make_line(10)
    # Perturb positions slightly
    nodes = nodes.__replace__(position=nodes.position + jnp.ones_like(nodes.position) * 0.3)
    nodes, edges = _run_steps(nodes, edges, n=30)

    lens = _seg_lengths(nodes, edges)
    err = jnp.abs(lens - edges.rest_lengths).max()
    assert err < 0.15, f"max length error {err:.4f} exceeds threshold"


# ──────────────────────────────────────────────────────────
# 3. Bending constraint
# ──────────────────────────────────────────────────────────

def test_bending_converges_to_rest_angle():
    """
    5-node line with bending_rest_angles = π/4 everywhere.
    Starting from all-zero thetas, angles should converge toward π/4.
    """
    n = 5
    rest_length = 2.0
    rest_angle = jnp.pi / 4

    nodes, edges = _make_line(n, rest_length)
    # Override rest angles
    edges = edges.__replace__(
        bending_rest_angles=jnp.full(edges.bending_pairs.shape[0], rest_angle),
    )

    nodes, edges = _run_steps(nodes, edges, n=50)

    diffs = _bend_diffs(nodes, edges)
    err = jnp.abs(diffs - rest_angle).mean()
    assert err < 0.3, f"bending did not converge: mean angle err = {err:.4f} rad"


# ──────────────────────────────────────────────────────────
# 4. L-shape: single 90° bend
# ──────────────────────────────────────────────────────────

def test_l_shape_bends():
    """
    3-node chain, rest_angle = π/2 at the middle hinge.
    Starting straight, the chain must develop a clear bend after simulation.
    """
    n = 3
    rest_length = 3.0
    nodes, edges = _make_line(n, rest_length)
    edges = edges.__replace__(
        bending_rest_angles=jnp.array([jnp.pi / 2]),
        bending_stiffness=jnp.array([1.0]),
    )

    nodes, edges = _run_steps(nodes, edges, n=100)

    # The theta difference should be closer to π/2 than to 0
    diff = _bend_diffs(nodes, edges)[0]
    assert abs(diff) > 0.3, f"no bend developed: theta diff = {diff:.4f} rad"
