from conftest import create_dummy_nodes
from dataclasses import replace

import jax.numpy as jnp
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.fields import Fields
from microcosmos.simulate import step
from microcosmos.solver.config import PBD_SCHEME


def _legacy_constraint(iteration, carry, config, grid_shape):
    del iteration, config, grid_shape
    return carry


def test_simulate():
    nodes, edges = create_dummy_nodes(5, shape=(10, 10))

    h, w = (10, 10)
    weights = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    f_grid = jnp.zeros((9, h, w))
    f_grid = f_grid.at[:].set(weights[:, None, None])

    fields = Fields(
        grid_shape=(h, w),
        steric=jnp.zeros((h, w)),
        fluid_velocity=jnp.zeros((2, h, w)),
        f_grid=f_grid,
    )
    dt = 0.01

    new_nodes, new_edges, new_fields = step(nodes, edges, fields, dt, PBD_SCHEME)

    assert isinstance(new_nodes, Nodes)
    assert new_nodes.position.shape == nodes.position.shape
    assert isinstance(new_fields, Fields)


def test_legacy_custom_constraint_does_not_require_mask_argument():
    nodes, edges = create_dummy_nodes(5, shape=(10, 10))
    fields = Fields(
        grid_shape=(10, 10),
        steric=jnp.zeros((10, 10)),
        fluid_velocity=jnp.zeros((2, 10, 10)),
        f_grid=jnp.ones((9, 10, 10)),
    )
    solver = replace(
        PBD_SCHEME,
        bending_constraint=_legacy_constraint,
        cycles_per_step=1,
        enable_fluid=False,
    )
    new_nodes, new_edges, _ = step(nodes, edges, fields, 0.01, solver)
    assert new_nodes.position.shape == nodes.position.shape
    assert new_edges.pairs.shape == edges.pairs.shape
