import jax.numpy as jnp
from microcosmos.forces import update_steric_potential
from microcosmos.structs.fields import Fields
from microcosmos.structs.nodes import Nodes


def test_update_steric_potential():
    shape = (10, 10)
    nodes = Nodes(
        position=jnp.array([[5.0, 5.0]]),
        velocity=jnp.zeros((1, 2)),
        color_bend=jnp.zeros(1),
        debug_vector=jnp.zeros((1, 2)),
    )

    h, w = shape
    weights = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    f_grid = jnp.zeros((9, h, w))
    f_grid = f_grid.at[:].set(weights[:, None, None])

    fields = Fields(
        grid_shape=shape,
        steric=jnp.zeros(shape),
        fluid_velocity=jnp.zeros((2, h, w)),
        f_grid=f_grid,
    )

    fields = update_steric_potential(nodes, fields)

    # Check if potential is updated around the position
    assert fields.steric[5, 5] > 0
    assert fields.steric[5, 6] > 0
