import jax.numpy as jnp
import numpy as np
from microcosmos.graph import make_edges, initialize_positions, compute_bending_pairs
import jax


class TestLoopEdges:
    """Tests for loop edge topology."""

    def test_loop_num_edges(self):
        """A loop of N nodes should have N edges."""
        num_nodes = 10
        edges, dep = make_edges(num_nodes, graph="loop")
        assert edges.shape[0] == num_nodes
        assert edges.shape[1] == 2

    def test_loop_every_node_appears(self):
        """Every node should appear in at least one edge."""
        num_nodes = 10
        edges, _ = make_edges(num_nodes, graph="loop")
        all_nodes = set(range(num_nodes))
        nodes_in_edges = set(np.array(edges[:, 0]).tolist()) | set(np.array(edges[:, 1]).tolist())
        assert all_nodes == nodes_in_edges

    def test_loop_edges_are_sequential(self):
        """In a loop, edge i should connect node i to node (i+1) % N."""
        num_nodes = 10
        edges, _ = make_edges(num_nodes, graph="loop")
        for i in range(num_nodes):
            assert edges[i, 0] == i
            assert edges[i, 1] == (i + 1) % num_nodes

    def test_loop_departure_angles_zero(self):
        """All departure angles in a loop should be 0."""
        num_nodes = 10
        _, dep = make_edges(num_nodes, graph="loop")
        assert jnp.allclose(dep, 0.0)

    def test_loop_wraps_around(self):
        """Last edge should connect last node to first."""
        num_nodes = 10
        edges, _ = make_edges(num_nodes, graph="loop")
        assert edges[-1, 0] == num_nodes - 1
        assert edges[-1, 1] == 0


class TestLineEdges:
    """Tests for line edge topology."""

    def test_line_num_edges(self):
        """A line of N nodes should have N-1 edges."""
        num_nodes = 10
        edges, _ = make_edges(num_nodes, graph="line")
        assert edges.shape[0] == num_nodes - 1

    def test_line_endpoints(self):
        """First node appears as source once, last node appears as target once."""
        num_nodes = 10
        edges, _ = make_edges(num_nodes, graph="line")
        sources = np.array(edges[:, 0])
        targets = np.array(edges[:, 1])

        # Node 0 appears as source exactly once
        assert np.sum(sources == 0) == 1
        # Node 0 never appears as target
        assert np.sum(targets == 0) == 0

        # Last node appears as target exactly once
        assert np.sum(targets == num_nodes - 1) == 1
        # Last node never appears as source
        assert np.sum(sources == num_nodes - 1) == 0

    def test_line_middle_nodes(self):
        """Middle nodes appear once as source and once as target."""
        num_nodes = 10
        edges, _ = make_edges(num_nodes, graph="line")
        sources = np.array(edges[:, 0])
        targets = np.array(edges[:, 1])

        for i in range(1, num_nodes - 1):
            assert np.sum(sources == i) == 1, f"Node {i} should appear as source once"
            assert np.sum(targets == i) == 1, f"Node {i} should appear as target once"

    def test_line_departure_angles_zero(self):
        """All departure angles in a line should be 0."""
        num_nodes = 10
        _, dep = make_edges(num_nodes, graph="line")
        assert jnp.allclose(dep, 0.0)


class TestTreeEdges:
    """Tests for tree edge topology."""

    def test_tree_num_edges(self):
        """A tree with N nodes should have N-1 edges."""
        num_nodes = 50
        edges, _ = make_edges(num_nodes, graph="tree")
        assert edges.shape[0] == num_nodes - 1

    def test_tree_junction_has_three_edges(self):
        """Junction node should have 3 incident edges (2 trunk + 1 branch each)."""
        num_nodes = 50
        edges, _ = make_edges(num_nodes, graph="tree")

        trunk_size = int(num_nodes * 0.6)
        junction_idx = trunk_size // 2

        # Count edges incident to junction
        edges_np = np.array(edges)
        incident = np.sum(edges_np[:, 0] == junction_idx) + np.sum(edges_np[:, 1] == junction_idx)
        # Junction has: trunk_prev->junction, junction->trunk_next, junction->branch1, junction->branch2
        # As source: junction->trunk_next, junction->branch1, junction->branch2 = 3
        # As target: trunk_prev->junction = 1
        # Total incident = 4
        assert incident >= 3, f"Junction should have at least 3 incident edges, got {incident}"

    def test_tree_branch_departure_angles(self):
        """Branch edges should have non-zero departure angles."""
        num_nodes = 50
        _, dep = make_edges(num_nodes, graph="tree")

        # There should be at least 2 edges with non-zero departure angles (branch connections)
        nonzero_dep = jnp.sum(jnp.abs(dep) > 0.1)
        assert nonzero_dep >= 2, f"Expected at least 2 branch edges, got {nonzero_dep}"


class TestMultipleInstances:
    """Tests for multiple independent instances."""

    def test_multiple_loops_edge_count(self):
        """Multiple loops: total edges = sum of individual loop edges."""
        num_nodes = 20
        num_instances = 2
        edges, _ = make_edges(num_nodes, graph="loop", num_instances=num_instances)
        # 10 nodes per instance = 10 edges per loop = 20 total
        assert edges.shape[0] == num_nodes

    def test_multiple_loops_independent(self):
        """Edges should not cross instance boundaries."""
        num_nodes = 20
        num_instances = 2
        edges, _ = make_edges(num_nodes, graph="loop", num_instances=num_instances)

        edges_np = np.array(edges)
        # First instance: nodes 0-9, second: 10-19
        for e_idx in range(edges_np.shape[0]):
            src, tgt = int(edges_np[e_idx, 0]), int(edges_np[e_idx, 1])
            # Both should be in the same instance
            src_instance = 0 if src < 10 else 1
            tgt_instance = 0 if tgt < 10 else 1
            assert src_instance == tgt_instance, \
                f"Edge ({src}, {tgt}) crosses instance boundary"

    def test_multiple_lines(self):
        """Multiple lines: each instance has N/instances - 1 edges."""
        num_nodes = 20
        num_instances = 2
        edges, _ = make_edges(num_nodes, graph="line", num_instances=num_instances)
        # 10 nodes per instance = 9 edges per line = 18 total
        assert edges.shape[0] == num_nodes - num_instances


class TestBendingPairs:
    """Tests for bending pair computation."""

    def test_line_bending_pairs_count(self):
        """A line of N nodes should have N-2 bending pairs."""
        num_nodes = 10
        edges, dep = make_edges(num_nodes, graph="line")
        bp, bp_rest = compute_bending_pairs(edges, dep, num_nodes)
        assert bp.shape[0] == num_nodes - 2
        assert bp.shape[1] == 2

    def test_loop_bending_pairs_count(self):
        """A loop of N nodes should have N bending pairs."""
        num_nodes = 10
        edges, dep = make_edges(num_nodes, graph="loop")
        bp, bp_rest = compute_bending_pairs(edges, dep, num_nodes)
        assert bp.shape[0] == num_nodes

    def test_tree_bending_pairs_at_junction(self):
        """A tree junction (1 in, 3 out) should produce 3 bending pairs at that node."""
        num_nodes = 50
        edges, dep = make_edges(num_nodes, graph="tree")
        bp, bp_rest = compute_bending_pairs(edges, dep, num_nodes)

        trunk_size = int(num_nodes * 0.6)
        junction_idx = trunk_size // 2

        # Count bending pairs that share the junction node
        # These are pairs (e_in, e_out) where e_in's target or e_out's source is the junction
        edges_np = np.array(edges)
        bp_np = np.array(bp)

        junction_bp_count = 0
        for b in range(bp_np.shape[0]):
            e_in, e_out = int(bp_np[b, 0]), int(bp_np[b, 1])
            # The shared node is: edges[e_in, 1] == edges[e_out, 0]
            shared = int(edges_np[e_in, 1])
            if shared == junction_idx:
                junction_bp_count += 1

        # Junction has 1 incoming trunk edge and 3 outgoing (trunk + 2 branches) → 3 bending pairs
        assert junction_bp_count == 3, f"Expected 3 bending pairs at junction, got {junction_bp_count}"

    def test_line_bending_rest_angles_zero(self):
        """Line bending rest angles should be 0 (no departure angle differences)."""
        num_nodes = 10
        edges, dep = make_edges(num_nodes, graph="line")
        bp, bp_rest = compute_bending_pairs(edges, dep, num_nodes)
        assert jnp.allclose(bp_rest, 0.0)

    def test_tree_junction_bending_rest_angles(self):
        """Junction bending pairs should have non-zero rest angles from departure angles."""
        num_nodes = 50
        edges, dep = make_edges(num_nodes, graph="tree")
        bp, bp_rest = compute_bending_pairs(edges, dep, num_nodes)

        # There should be bending pairs with non-zero rest angles (at junctions)
        nonzero = jnp.sum(jnp.abs(bp_rest) > 0.1)
        assert nonzero >= 2, f"Expected at least 2 non-zero bending rest angles, got {nonzero}"



class TestInitializePositions:
    """Tests for position initialization."""
    def test_initialize_positions_multiple_loops_within_bounds(self):
        """Test that multiple loop instances positions are within expected bounds."""
        num_nodes = 20
        num_instances = 2
        shape = jnp.array([100.0, 100.0])
        key = jax.random.PRNGKey(0)

        positions = initialize_positions(num_nodes, "loop", shape, key, num_instances)

        assert jnp.all(positions >= 0), f"Some positions are negative: min={jnp.min(positions)}"
        assert jnp.all(positions[:, 0] <= shape[0]), f"Some x positions exceed bounds"
        assert jnp.all(positions[:, 1] <= shape[1]), f"Some y positions exceed bounds"

    def test_initialize_positions_all_types_within_bounds(self):
        """Test that all graph types produce positions within bounds."""
        num_nodes = 50
        shape = jnp.array([100.0, 100.0])

        for graph_type in ["loop", "line", "tree"]:
            key = jax.random.PRNGKey(42)
            positions = initialize_positions(num_nodes, graph_type, shape, key)

            assert jnp.all(positions >= 0), f"{graph_type}: Some positions are negative"
            assert jnp.all(positions[:, 0] <= shape[0]), f"{graph_type}: Some x positions exceed bounds"
            assert jnp.all(positions[:, 1] <= shape[1]), f"{graph_type}: Some y positions exceed bounds"
