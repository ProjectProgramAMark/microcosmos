"""
Tests for NetworkX-based graph builder with edge-centric API.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from microcosmos.graph_builder import GraphBuilder


def test_simple_loop():
    """Test creating a simple loop topology."""
    builder = GraphBuilder()
    loop = builder.add_loop(10)

    edges, departure_angles = builder.build()

    assert edges.shape == (10, 2)  # loop of 10 = 10 edges
    assert departure_angles.shape == (10,)
    assert loop == list(range(10))

    # All departure angles should be 0 for a loop
    assert jnp.allclose(departure_angles, 0.0)

    # Each node should appear as source and target exactly once
    sources = np.array(edges[:, 0])
    targets = np.array(edges[:, 1])
    assert set(sources.tolist()) == set(range(10))
    assert set(targets.tolist()) == set(range(10))


def test_simple_line():
    """Test creating a simple line topology."""
    builder = GraphBuilder()
    line = builder.add_line(15)

    edges, departure_angles = builder.build()

    assert edges.shape == (14, 2)  # line of 15 = 14 edges
    assert departure_angles.shape == (14,)

    # All departure angles should be 0 for a line
    assert jnp.allclose(departure_angles, 0.0)

    # End nodes: node 0 only appears as source, node 14 only as target
    sources = np.array(edges[:, 0])
    targets = np.array(edges[:, 1])
    assert np.sum(sources == 0) == 1
    assert np.sum(targets == 0) == 0
    assert np.sum(targets == 14) == 1
    assert np.sum(sources == 14) == 0


def test_loop_with_three_branches():
    """Test the example: loop with 3 branches."""
    builder = GraphBuilder()

    # Create loop of 50 nodes
    loop = builder.add_loop(50)

    # Add three branches at nodes 1, 10, 20, each 40 nodes long
    branch1 = builder.add_branch(from_node=1, length=40)
    branch2 = builder.add_branch(from_node=10, length=40)
    branch3 = builder.add_branch(from_node=20, length=40)

    edges, departure_angles = builder.build()

    # Total nodes: 50 (loop) + 3*40 (branches) = 170
    # Total edges: 50 (loop) + 3*39 (branch internal) + 3 (junction) = 170
    total_edges = 50 + 3 * 39 + 3
    assert edges.shape == (total_edges, 2)

    edges_np = np.array(edges)

    # Branch attachment points should have 3 outgoing edges (next in loop + branch start)
    # or be incident to 3 edges total
    for attachment_node in [1, 10, 20]:
        incident = np.sum(edges_np[:, 0] == attachment_node) + np.sum(edges_np[:, 1] == attachment_node)
        assert incident >= 3, f"Node {attachment_node} should have at least 3 incident edges, got {incident}"


def test_tree_topology():
    """Test creating a tree with trunk and two branches."""
    builder = GraphBuilder()

    # Create trunk (vertical line)
    trunk = builder.add_line(40)

    # Add two branches from the middle
    branch_point = trunk[20]
    branch1 = builder.add_branch(from_node=branch_point, length=20)
    branch2 = builder.add_branch(from_node=branch_point, length=20)

    edges, departure_angles = builder.build()

    # Total: 40 + 20 + 20 = 80 nodes
    # Edges: 39 (trunk) + 19 (branch1 internal) + 19 (branch2 internal) + 2 (junctions) = 79
    assert edges.shape == (79, 2)

    edges_np = np.array(edges)

    # Branch point should have 4 incident edges (prev in trunk, next in trunk, 2 branches)
    incident = np.sum(edges_np[:, 0] == branch_point) + np.sum(edges_np[:, 1] == branch_point)
    assert incident == 4, f"Branch point should have 4 incident edges, got {incident}"

    # All branch edges (junction + internal) should have non-zero departure angles
    # 2 junction edges + 19 branch1 internal + 19 branch2 internal = 40
    dep_np = np.array(departure_angles)
    nonzero_dep = np.sum(np.abs(dep_np) > 0.1)
    assert nonzero_dep == 40, f"Expected 40 branch edges with non-zero departure, got {nonzero_dep}"


def test_multiple_loops():
    """Test creating multiple disconnected loops."""
    builder = GraphBuilder()

    loop1 = builder.add_loop(10)
    loop2 = builder.add_loop(15)
    loop3 = builder.add_loop(8)

    edges, departure_angles = builder.build()

    # Total edges: 10 + 15 + 8 = 33
    assert edges.shape == (33, 2)
    assert len(loop1) == 10
    assert len(loop2) == 15
    assert len(loop3) == 8

    # Verify loops are separate (check node IDs)
    assert loop1 == list(range(0, 10))
    assert loop2 == list(range(10, 25))
    assert loop3 == list(range(25, 33))

    # No edges should cross loop boundaries
    edges_np = np.array(edges)
    for e_idx in range(edges_np.shape[0]):
        src, tgt = int(edges_np[e_idx, 0]), int(edges_np[e_idx, 1])
        if src < 10:
            assert tgt < 10, f"Edge ({src}, {tgt}) crosses loop boundary"
        elif src < 25:
            assert 10 <= tgt < 25, f"Edge ({src}, {tgt}) crosses loop boundary"
        else:
            assert tgt >= 25, f"Edge ({src}, {tgt}) crosses loop boundary"


def test_complex_branching():
    """Test trunk with branches that have sub-branches."""
    builder = GraphBuilder()

    # Main trunk
    trunk = builder.add_line(30)

    # Primary branch from middle of trunk
    primary_branch = builder.add_branch(from_node=trunk[15], length=20)

    # Secondary branches from the primary branch
    sub_branch1 = builder.add_branch(from_node=primary_branch[10], length=10)
    sub_branch2 = builder.add_branch(from_node=primary_branch[10], length=10)

    edges, departure_angles = builder.build()

    # Total: 30 + 20 + 10 + 10 = 70 nodes
    # Edges: 29 + 19 + 9 + 9 + 3 (junctions) = 69
    assert edges.shape == (69, 2)

    edges_np = np.array(edges)

    # The sub-branch point should have 4 incident edges
    sub_branch_point = primary_branch[10]
    incident = np.sum(edges_np[:, 0] == sub_branch_point) + np.sum(edges_np[:, 1] == sub_branch_point)
    assert incident == 4


def test_branch_departure_angles():
    """Test that branch edges get correct departure angles (pi/2 and -pi/2)."""
    builder = GraphBuilder()
    trunk = builder.add_line(20)
    branch1 = builder.add_branch(from_node=trunk[10], length=10)
    branch2 = builder.add_branch(from_node=trunk[10], length=10)

    edges, departure_angles = builder.build()

    dep_np = np.array(departure_angles)
    edges_np = np.array(edges)

    # Find the two junction edges (from trunk[10] to branch starts)
    junction_mask = edges_np[:, 0] == trunk[10]
    junction_deps = dep_np[junction_mask]

    # Chain edge (trunk[10] -> trunk[11]) should have departure 0
    # Branch edges should have +pi/2 and -pi/2
    nonzero = junction_deps[np.abs(junction_deps) > 0.1]
    assert len(nonzero) == 2
    assert np.isclose(np.abs(nonzero[0]), np.pi / 2, atol=0.01)
    assert np.isclose(np.abs(nonzero[1]), np.pi / 2, atol=0.01)
    # One positive, one negative
    assert np.sign(nonzero[0]) != np.sign(nonzero[1])


def test_empty_graph():
    """Test building an empty graph."""
    builder = GraphBuilder()
    edges, departure_angles = builder.build()

    assert edges.shape == (0, 2)
    assert departure_angles.shape == (0,)


def test_non_contiguous_nodes_error():
    """Test that an error is raised for non-contiguous node IDs."""
    builder = GraphBuilder()
    builder.G.add_edge(0, 1, departure_angle=0.0)
    builder.G.add_edge(1, 5, departure_angle=0.0)  # Gap in node IDs
    builder.next_id = 6

    with pytest.raises(ValueError, match="contiguous integers"):
        builder.build()
