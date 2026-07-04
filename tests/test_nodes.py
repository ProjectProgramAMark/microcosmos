import jax.numpy as jnp
from microcosmos.structs.nodes import Nodes


def test_nodes_initialization():
    num_nodes = 10
    shape = jnp.array([100, 100])

    nodes = Nodes(
        position=jnp.zeros((num_nodes, 2)),
        velocity=jnp.zeros((num_nodes, 2)),
        color_bend=jnp.zeros(num_nodes),
        debug_vector=jnp.zeros((num_nodes, 2)),
    )

    assert len(nodes) == num_nodes

    # Test slicing - all fields are N-shaped, all get sliced
    subset = nodes[:5]
    assert len(subset) == 5
    assert jnp.array_equal(subset.position, nodes.position[:5])
