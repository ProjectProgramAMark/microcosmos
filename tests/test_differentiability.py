import jax
import jax.numpy as jnp
from conftest import create_dummy_nodes

from microcosmos.simulate import simulate
from microcosmos.solver.config import PBD_SCHEME
from microcosmos.structs.fields import Fields

def forward(nodes, edges, dt, num_steps, grid_shape=(100, 100)):
    h, w = grid_shape
    weights = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    f_grid = jnp.zeros((9, h, w))
    f_grid = f_grid.at[:].set(weights[:, None, None])

    fields = Fields(
        grid_shape=grid_shape,
        steric=jnp.zeros(grid_shape),
        fluid_velocity=jnp.zeros((2, h, w)),
        f_grid=f_grid,
    )
    nodes_timeseries, _ = simulate(
        nodes,
        edges,
        fields,
        dt,
        num_steps,
        PBD_SCHEME,
    )
    return nodes_timeseries[-1]


def test_differentiability():
    """Test that the simulation is differentiable and produces valid gradients."""
    num_nodes = 10
    nodes, edges = create_dummy_nodes(num_nodes)
    dt = 0.01
    num_steps = 1

    # Run simulation
    final_nodes = forward(nodes, edges, dt, num_steps)

    def loss_fn(final_nodes):
        return -jnp.sum(final_nodes.position ** 2)

    loss, grads = jax.value_and_grad(loss_fn, allow_int=True)(final_nodes)

    assert loss is not None
    assert jnp.isfinite(loss)
    assert grads.position.shape == final_nodes.position.shape
    assert jnp.all(jnp.isfinite(grads.position))

def test_gradient_through_simulation():
    """Test that gradients can be computed through the entire simulation."""
    num_nodes = 10
    nodes, edges = create_dummy_nodes(num_nodes)
    dt = 0.01
    num_steps = 1
    params = {"bending_rest_angles": edges.bending_rest_angles, "rest_lengths": edges.rest_lengths}

    def loss_fn(params):
        updated_edges = edges.__replace__(**params)
        final_nodes = forward(nodes, updated_edges, dt, num_steps)
        return jnp.mean(final_nodes.position ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    assert jnp.isfinite(loss)
    assert grads['bending_rest_angles'].shape == edges.bending_rest_angles.shape

    # print grad norm
    print("Gradient norm:", jnp.linalg.norm(grads['bending_rest_angles']))

    assert jnp.all(jnp.isfinite(grads['bending_rest_angles']))
    assert jnp.all(jnp.isfinite(grads['rest_lengths']))
