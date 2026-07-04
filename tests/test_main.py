import jax.numpy as jnp
import numpy as np
from microcosmos.main import make_graph


def test_make_graph_loop():
    num_nodes = 5
    edges, departure_angles = make_graph(num_nodes, graph_type="loop")

    # Loop of 5 nodes = 5 edges
    assert edges.shape == (5, 2)

    # Each edge connects i -> (i+1) % 5
    for i in range(5):
        assert edges[i, 0] == i
        assert edges[i, 1] == (i + 1) % 5

    # All departure angles should be 0
    assert jnp.allclose(departure_angles, 0.0)


def test_make_graph_line():
    num_nodes = 5
    edges, departure_angles = make_graph(num_nodes, graph_type="line")

    # Line of 5 nodes = 4 edges
    assert edges.shape == (4, 2)

    # Each edge connects i -> i+1
    for i in range(4):
        assert edges[i, 0] == i
        assert edges[i, 1] == i + 1

    # All departure angles should be 0
    assert jnp.allclose(departure_angles, 0.0)

    # Node 0 only appears as source, node 4 only as target
    edges_np = np.array(edges)
    assert np.sum(edges_np[:, 0] == 0) == 1
    assert np.sum(edges_np[:, 1] == 0) == 0
    assert np.sum(edges_np[:, 1] == 4) == 1
    assert np.sum(edges_np[:, 0] == 4) == 0
