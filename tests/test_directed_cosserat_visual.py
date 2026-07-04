"""
Visual tests for build_directed_cosserat_graph.

Run with: uv run pytest tests/test_directed_cosserat_visual.py -v
Images saved to tests/visual_output/
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pytest

from microcosmos.graph import build_directed_cosserat_graph, validate_rod_topology

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "visual_output")


@pytest.fixture(autouse=True)
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_directed_cosserat(G_undirected, directed_G, root_node, title, filename):
    """Draw the original undirected graph and the directed result side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Spring layout with high iterations for a clean spread
    pos = nx.spring_layout(G_undirected, seed=42, k=3.0 / (G_undirected.number_of_nodes() ** 0.3),
                           iterations=10000)

    # --- Left: Original undirected graph ---
    ax = axes[0]
    ax.set_title(f"{title}\n(Original Undirected)", fontsize=13, fontweight="bold")
    nx.draw_networkx_nodes(G_undirected, pos, ax=ax, node_color="#a8d5e2",
                           node_size=500, edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G_undirected, pos, ax=ax, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G_undirected, pos, ax=ax, width=1.5, edge_color="#888888",
                           style="solid")
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Right: Directed Cosserat graph ---
    ax = axes[1]
    ax.set_title(f"{title}\n(Directed Cosserat)", fontsize=13, fontweight="bold")

    # Classify nodes: root, leaves (sinks), interior
    out_degrees = dict(directed_G.out_degree())
    in_degrees = dict(directed_G.in_degree())
    node_colors = []
    for n in directed_G.nodes():
        if n == root_node:
            node_colors.append("#e74c3c")      # red root
        elif out_degrees[n] == 0:
            node_colors.append("#f39c12")      # orange leaf/sink
        elif out_degrees[n] >= 2:
            node_colors.append("#9b59b6")      # purple junction
        else:
            node_colors.append("#2ecc71")      # green interior

    nx.draw_networkx_nodes(directed_G, pos, ax=ax, node_color=node_colors,
                           node_size=500, edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(directed_G, pos, ax=ax, font_size=10, font_weight="bold")

    # Draw directed edges with prominent arrows
    nx.draw_networkx_edges(
        directed_G, pos, ax=ax,
        width=2.5,
        edge_color="#2c3e50",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=25,
        connectionstyle="arc3,rad=0.08",
        min_source_margin=17,
        min_target_margin=17,
    )

    # Legend
    legend_items = [
        mpatches.Patch(color="#e74c3c", label=f"Root (node {root_node})"),
        mpatches.Patch(color="#2ecc71", label="Interior (1-in, 1-out)"),
        mpatches.Patch(color="#9b59b6", label="Junction (out-degree >= 2)"),
        mpatches.Patch(color="#f39c12", label="Leaf / Sink (out-degree 0)"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def make_cycle_dumbbell(loop1_size=6, bridge_size=3, loop2_size=6):
    """Build a dumbbell from two simple cycles connected by a bridge chain.

    Returns an undirected graph with node layout:
      loop1: 0 .. loop1_size-1  (cycle)
      bridge: loop1_size .. loop1_size+bridge_size-1  (chain)
      loop2: loop1_size+bridge_size .. end  (cycle)
    The bridge connects loop1 node 0 to loop2 node 0.
    """
    G = nx.Graph()
    # Loop 1
    for i in range(loop1_size):
        G.add_edge(i, (i + 1) % loop1_size)
    # Bridge: loop1[0] -> b0 -> b1 -> ... -> loop2[0]
    b_start = loop1_size
    prev = 0  # attach to loop1 node 0
    for i in range(bridge_size):
        G.add_edge(prev, b_start + i)
        prev = b_start + i
    # Loop 2
    l2_start = b_start + bridge_size
    for i in range(loop2_size):
        G.add_edge(l2_start + i, l2_start + (i + 1) % loop2_size)
    # Connect bridge end to loop2 node 0
    G.add_edge(prev, l2_start)
    return G


class TestDumbbellDirected:
    """Visual test: dumbbell graph (cycle-bridge-cycle)."""

    def test_dumbbell(self):
        G = make_cycle_dumbbell(loop1_size=6, bridge_size=3, loop2_size=6)
        directed_G, root = build_directed_cosserat_graph(G)

        # Every undirected edge gets exactly one directed edge
        assert directed_G.number_of_edges() == G.number_of_edges()
        assert directed_G.number_of_nodes() == G.number_of_nodes()

        draw_directed_cosserat(G, directed_G, root, "Dumbbell (6-3-6)", "dumbbell.png")


class TestDumbbellWithBranches:
    """Visual test: dumbbell with branches from loop, bridge, and other loop."""

    def test_dumbbell_with_branches(self):
        G = make_cycle_dumbbell(loop1_size=6, bridge_size=3, loop2_size=6)
        num = G.number_of_nodes()  # 15

        # Branch from loop1 (node 3 — mid-loop)
        G.add_edge(3, num)
        G.add_edge(num, num + 1)
        G.add_edge(num + 1, num + 2)

        # Branch from bridge (node 7 — middle of bridge)
        G.add_edge(7, num + 3)
        G.add_edge(num + 3, num + 4)

        # Branch from loop2 (node 12 — mid-loop)
        G.add_edge(12, num + 5)
        G.add_edge(num + 5, num + 6)
        G.add_edge(num + 6, num + 7)

        directed_G, root = build_directed_cosserat_graph(G)

        assert directed_G.number_of_edges() == G.number_of_edges()
        assert directed_G.number_of_nodes() == G.number_of_nodes()

        draw_directed_cosserat(
            G, directed_G, root,
            "Dumbbell + Branches\n(loop1@3, bridge@7, loop2@12)",
            "dumbbell_branches.png",
        )


class TestFlowerGraph:
    """Visual test: 4 petal rings radiating from a hub via bridge edges.

    Each petal is its own biconnected component (simple cycle) attached to
    the hub by a single bridge edge.  Bridge removal isolates each ring
    cleanly, so find_cycle traces each one perfectly.
    Pushes: high-degree hub (degree 4), many biconnected components all
    meeting at a single node.
    """

    def test_flower(self):
        G = nx.Graph()
        hub = 0
        node_id = 1
        petal_sizes = [5, 7, 6, 4]  # asymmetric

        for petal_size in petal_sizes:
            # Ring nodes
            ring_start = node_id
            for j in range(petal_size):
                G.add_edge(ring_start + j, ring_start + (j + 1) % petal_size)
            # Single bridge edge: hub -> first node of this ring
            G.add_edge(hub, ring_start)
            node_id += petal_size

        directed_G, root = build_directed_cosserat_graph(G)

        assert directed_G.number_of_edges() == G.number_of_edges()
        assert directed_G.number_of_nodes() == G.number_of_nodes()

        draw_directed_cosserat(
            G, directed_G, root,
            "Flower (4 rings off hub via bridges)",
            "flower.png",
        )


class TestChainOfRingsWithTail:
    """Visual test: 4 rings daisy-chained by bridge segments + forked tail.

    Each ring is a simple cycle.  Bridges are short chains (2 nodes each)
    connecting a mid-ring node to the first node of the next ring.
    A forked branch hangs off the last ring.
    Pushes: deep sequential biconnected components, many bridge/loop
    boundaries, branching at the far end.
    """

    def test_chain_of_rings(self):
        G = nx.Graph()
        node_id = 0
        ring_sizes = [5, 6, 5, 7]
        ring_mids = []   # mid-node of each ring (bridge attachment point)

        for i, rsize in enumerate(ring_sizes):
            ring_start = node_id
            # Build the cycle
            for j in range(rsize):
                G.add_edge(ring_start + j, ring_start + (j + 1) % rsize)
            ring_mids.append(ring_start + rsize // 2)
            node_id += rsize

            # 2-node bridge to the NEXT ring (connects mid of this ring
            # to the first node of the next ring, which doesn't exist yet)
            if i < len(ring_sizes) - 1:
                b0, b1 = node_id, node_id + 1
                G.add_edge(ring_mids[-1], b0)
                G.add_edge(b0, b1)
                # b1 will connect to the next ring's first node below
                node_id += 2

        # Stitch bridge ends into the next ring's first node
        # ring 0 mid -> b0 -> b1 -> ring1[0],  ring1 mid -> b2 -> b3 -> ring2[0], ...
        bridge_idx = 0
        running = 0
        for i in range(len(ring_sizes) - 1):
            running += ring_sizes[i]
            b1 = running + 1  # second bridge node
            next_ring_start = running + 2
            G.add_edge(b1, next_ring_start)
            running = next_ring_start + ring_sizes[i + 1] - ring_sizes[i + 1]
            # (running is updated by the loop above via node_id, but we
            #  just need to connect b1 -> next_ring_start which we know)

        # Forked tail off the last ring
        last_mid = ring_mids[-1]
        tail_start = node_id
        # Stem (3 edges)
        G.add_edge(last_mid, tail_start)
        G.add_edge(tail_start, tail_start + 1)
        G.add_edge(tail_start + 1, tail_start + 2)
        fork = tail_start + 2
        node_id = fork + 1
        # Two prongs off the fork
        G.add_edge(fork, node_id)
        G.add_edge(node_id, node_id + 1)
        node_id += 2
        G.add_edge(fork, node_id)
        node_id += 1

        directed_G, root = build_directed_cosserat_graph(G)

        assert directed_G.number_of_edges() == G.number_of_edges()
        assert directed_G.number_of_nodes() == G.number_of_nodes()

        draw_directed_cosserat(
            G, directed_G, root,
            "Chain of 4 Rings + Forked Tail",
            "chain_of_rings.png",
        )


class TestValidateRodTopology:
    """Tests that validate_rod_topology catches invalid inputs."""

    def test_rejects_digraph(self):
        with pytest.raises(TypeError, match="undirected"):
            validate_rod_topology(nx.DiGraph())

    def test_rejects_empty_graph(self):
        with pytest.raises(ValueError, match="empty"):
            validate_rod_topology(nx.Graph())

    def test_rejects_no_edges(self):
        G = nx.Graph()
        G.add_node(0)
        with pytest.raises(ValueError, match="no edges"):
            validate_rod_topology(G)

    def test_rejects_self_loop(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(1, 1)
        with pytest.raises(ValueError, match="self-loops"):
            validate_rod_topology(G)

    def test_rejects_complete_graph_k4(self):
        """K4 has biconnected components where nodes have degree 3."""
        with pytest.raises(ValueError, match="not a simple cycle"):
            validate_rod_topology(nx.complete_graph(4))

    def test_rejects_barbell_with_complete_ends(self):
        """nx.barbell_graph(5,1) uses K5 cliques — not simple cycles."""
        with pytest.raises(ValueError, match="not a simple cycle"):
            validate_rod_topology(nx.barbell_graph(5, 1))

    def test_rejects_petersen_graph(self):
        """Petersen graph has biconnected components with degree-3 nodes."""
        with pytest.raises(ValueError, match="not a simple cycle"):
            validate_rod_topology(nx.petersen_graph())

    def test_rejects_theta_graph(self):
        """Theta graph: 3 independent paths between nodes 0 and 3.

        All 6 nodes form one biconnected component where nodes 0 and 3
        have degree 3 — find_cycle would miss edges.
        """
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3),   # path 1
                          (0, 4), (4, 5), (5, 3),   # path 2
                          (0, 3)])                   # path 3 (direct)
        with pytest.raises(ValueError, match="not a simple cycle"):
            validate_rod_topology(G)

    def test_accepts_bowtie(self):
        """Bowtie (two triangles sharing a cut vertex) IS valid.

        Node 0 is an articulation point — each triangle is its own
        biconnected component, both simple cycles. Bridge removal
        handles this correctly.
        """
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        G.add_edges_from([(0, 3), (3, 4), (4, 0)])
        validate_rod_topology(G)  # should not raise

    def test_accepts_simple_cycle(self):
        validate_rod_topology(nx.cycle_graph(6))

    def test_accepts_simple_path(self):
        validate_rod_topology(nx.path_graph(5))

    def test_accepts_dumbbell(self):
        validate_rod_topology(make_cycle_dumbbell())


REJECTED_DIR = os.path.join(os.path.dirname(__file__), "visual_output", "rejected")


def draw_rejected(G, title, reason, filename):
    """Draw an invalid topology with the rejection reason annotated."""
    os.makedirs(REJECTED_DIR, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    pos = nx.spring_layout(G, seed=42, k=3.0 / (G.number_of_nodes() ** 0.3),
                           iterations=10000)

    # Highlight biconnected components that are invalid
    bad_nodes = set()
    for comp_nodes in nx.biconnected_components(G):
        sg = G.subgraph(comp_nodes)
        if sg.number_of_edges() > 1:
            if any(d != 2 for _, d in sg.degree()):
                bad_nodes.update(comp_nodes)

    node_colors = ["#e74c3c" if n in bad_nodes else "#a8d5e2"
                   for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=500, edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

    # Color edges inside bad bicomps red
    bad_edges = []
    ok_edges = []
    for u, v in G.edges():
        if u in bad_nodes and v in bad_nodes:
            bad_edges.append((u, v))
        else:
            ok_edges.append((u, v))

    nx.draw_networkx_edges(G, pos, edgelist=ok_edges, ax=ax,
                           width=1.5, edge_color="#888888")
    nx.draw_networkx_edges(G, pos, edgelist=bad_edges, ax=ax,
                           width=2.5, edge_color="#e74c3c", style="solid")

    ax.set_title(f"REJECTED: {title}", fontsize=14, fontweight="bold",
                 color="#e74c3c")

    # Reason text box at bottom
    ax.text(0.5, -0.05, reason, transform=ax.transAxes, fontsize=10,
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffeaa7", alpha=0.9))

    legend_items = [
        mpatches.Patch(color="#e74c3c", label="Invalid biconnected component"),
        mpatches.Patch(color="#a8d5e2", label="Valid nodes"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(REJECTED_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


class TestRejectedVisuals:
    """Generate visual output for topologies that fail validation."""

    def test_rejected_k4(self):
        G = nx.complete_graph(4)
        with pytest.raises(ValueError):
            validate_rod_topology(G)
        draw_rejected(G, "Complete Graph K4",
                      "Every node has degree 3 — not a simple cycle",
                      "rejected_k4.png")

    def test_rejected_barbell_complete(self):
        G = nx.barbell_graph(5, 1)
        with pytest.raises(ValueError):
            validate_rod_topology(G)
        draw_rejected(G, "Barbell (K5 cliques)",
                      "nx.barbell_graph uses complete subgraphs, not simple cycles",
                      "rejected_barbell_k5.png")

    def test_rejected_petersen(self):
        G = nx.petersen_graph()
        with pytest.raises(ValueError):
            validate_rod_topology(G)
        draw_rejected(G, "Petersen Graph",
                      "Single biconnected component, all nodes degree 3",
                      "rejected_petersen.png")

    def test_rejected_theta(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3),
                          (0, 4), (4, 5), (5, 3),
                          (0, 3)])
        with pytest.raises(ValueError):
            validate_rod_topology(G)
        draw_rejected(G, "Theta Graph",
                      "3 paths between nodes 0-3 in one bicomp — find_cycle misses edges",
                      "rejected_theta.png")

    def test_rejected_wheel(self):
        G = nx.wheel_graph(7)
        with pytest.raises(ValueError):
            validate_rod_topology(G)
        draw_rejected(G, "Wheel Graph (7 nodes)",
                      "Hub connects to all rim nodes — dense biconnected component",
                      "rejected_wheel.png")
