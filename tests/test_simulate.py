from conftest import create_dummy_nodes
import jax.numpy as jnp
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.fields import Fields
from microcosmos.simulate import step
from microcosmos.solver.config import PBD_SCHEME


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
