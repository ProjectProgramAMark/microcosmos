"""
Visual simulation tests for topologies built via build_directed_cosserat_graph.

Run with: uv run pytest tests/test_simulation_visual.py -v -s
Outputs saved to tests/outputs/
"""

import os

import pytest
import jax
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from microcosmos.graph import (
    build_simulation_from_undirected,
    validate_rod_topology,
    build_directed_cosserat_graph,
)
from microcosmos.simulate import simulate
from microcosmos.rendering import animate
from microcosmos.solver.config import PBD_SCHEME

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def draw_directed_cosserat(G_undirected, directed_G, root_node, title, filename):
    """Draw the original undirected graph and the directed result side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    n_nodes = G_undirected.number_of_nodes()
    pos = nx.spring_layout(
        G_undirected, seed=42,
        k=5.0 / (n_nodes ** 0.3),
        iterations=1000,
    )

    # Scale node size and labels with graph size
    node_sz = max(30, 600 // (1 + n_nodes // 10))
    font_sz = max(4, 10 - n_nodes // 20)

    # --- Left: Original undirected graph ---
    ax = axes[0]
    ax.set_title(f"{title}\n(Original Undirected)", fontsize=13, fontweight="bold")
    nx.draw_networkx_nodes(G_undirected, pos, ax=ax, node_color="#a8d5e2",
                           node_size=node_sz, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G_undirected, pos, ax=ax, font_size=font_sz)
    nx.draw_networkx_edges(G_undirected, pos, ax=ax, width=0.8, edge_color="#888888")
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Right: Directed Cosserat graph ---
    ax = axes[1]
    ax.set_title(f"{title}\n(Directed Cosserat)", fontsize=13, fontweight="bold")

    out_degrees = dict(directed_G.out_degree())
    in_degrees = dict(directed_G.in_degree())
    node_colors = []
    for n in directed_G.nodes():
        if n == root_node:
            node_colors.append("#e74c3c")      # red root
        elif out_degrees[n] == 0:
            node_colors.append("#f39c12")      # orange sink
        elif out_degrees[n] >= 2:
            node_colors.append("#9b59b6")      # purple: out-junction (fan-out)
        elif in_degrees[n] >= 2:
            node_colors.append("#3498db")      # blue: in-junction (fan-in)
        else:
            node_colors.append("#2ecc71")      # green interior

    nx.draw_networkx_nodes(directed_G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sz, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(directed_G, pos, ax=ax, font_size=font_sz)
    nx.draw_networkx_edges(
        directed_G, pos, ax=ax, width=1.2, edge_color="#2c3e50",
        arrows=True, arrowstyle="-|>", arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=max(4, node_sz ** 0.5),
        min_target_margin=max(4, node_sz ** 0.5),
    )

    legend_items = [
        mpatches.Patch(color="#e74c3c", label=f"Root (node {root_node})"),
        mpatches.Patch(color="#2ecc71", label="Interior (1-in, 1-out)"),
        mpatches.Patch(color="#9b59b6", label="Out-junction (out-degree >= 2)"),
        mpatches.Patch(color="#3498db", label="In-junction (in-degree >= 2)"),
        mpatches.Patch(color="#f39c12", label="Leaf / Sink (out-degree 0)"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def run_and_save(nodes, edges, fields, filename, num_steps=400, dt=0.01):
    """Run simulation and save animation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nodes_timeseries, fields_timeseries = simulate(
        nodes=nodes,
        edges=edges,
        fields=fields,
        num_steps=num_steps,
        dt=dt,
        solver_config=PBD_SCHEME,
    )
    jax.block_until_ready(nodes_timeseries)
    filepath = os.path.join(OUTPUT_DIR, filename)
    animate(
        nodes_timeseries, #
        fields_timeseries, # [:60]
        filename=filepath,
        subsample=3,
        # subsample=1,
        animate_fluid_velocity=False
    )
    print(f"Saved: {filepath}")


def make_dumbbell(loop1_size=40, bridge_size=20, loop2_size=40):
    """Two simple cycles connected by a bridge chain (~100 nodes)."""
    G = nx.Graph()
    # Loop 1
    for i in range(loop1_size):
        G.add_edge(i, (i + 1) % loop1_size)
    # Bridge
    b_start = loop1_size
    prev = 0
    for i in range(bridge_size):
        G.add_edge(prev, b_start + i)
        prev = b_start + i
    # Loop 2
    l2_start = b_start + bridge_size
    for i in range(loop2_size):
        G.add_edge(l2_start + i, l2_start + (i + 1) % loop2_size)
    # Connect bridge end to loop2
    G.add_edge(prev, l2_start)
    return G


def make_flower(petal_sizes=(4, 4, 4, 4), bridge_length=1):
    """Hub node with petal rings attached via bridge chains.

    Args:
        petal_sizes: Size of each petal ring.
        bridge_length: Number of edges in the bridge chain from hub to each petal.
            1 means a single direct edge (default).
    """
    G = nx.Graph()
    hub = 0
    node_id = 1
    for petal_size in petal_sizes:
        # Build bridge chain: hub -> b0 -> b1 -> ... -> ring_start
        prev = hub
        for _ in range(bridge_length - 1):
            G.add_edge(prev, node_id)
            prev = node_id
            node_id += 1
        ring_start = node_id
        G.add_edge(prev, ring_start)
        for j in range(petal_size):
            G.add_edge(ring_start + j, ring_start + (j + 1) % petal_size)
        node_id += petal_size
    return G


def make_chain_of_rings(ring_sizes=(5, 6, 5, 7)):
    """Rings daisy-chained by 2-node bridge segments + forked tail."""
    G = nx.Graph()
    node_id = 0
    ring_mids = []

    for i, rsize in enumerate(ring_sizes):
        ring_start = node_id
        for j in range(rsize):
            G.add_edge(ring_start + j, ring_start + (j + 1) % rsize)
        ring_mids.append(ring_start + rsize // 2)
        node_id += rsize

        if i < len(ring_sizes) - 1:
            b0, b1 = node_id, node_id + 1
            G.add_edge(ring_mids[-1], b0)
            G.add_edge(b0, b1)
            node_id += 2

    # Stitch bridge ends into next ring's first node
    running = 0
    for i in range(len(ring_sizes) - 1):
        running += ring_sizes[i]
        b1 = running + 1
        next_ring_start = running + 2
        G.add_edge(b1, next_ring_start)
        running = next_ring_start + ring_sizes[i + 1] - ring_sizes[i + 1]

    # Forked tail off last ring
    last_mid = ring_mids[-1]
    tail_start = node_id
    G.add_edge(last_mid, tail_start)
    G.add_edge(tail_start, tail_start + 1)
    G.add_edge(tail_start + 1, tail_start + 2)
    fork = tail_start + 2
    node_id = fork + 1
    G.add_edge(fork, node_id)
    G.add_edge(node_id, node_id + 1)
    node_id += 2
    G.add_edge(fork, node_id)
    return G


def make_bowtie():
    """Two triangles sharing a cut vertex (node 0)."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    G.add_edges_from([(0, 3), (3, 4), (4, 0)])
    return G


def make_dumbbell_with_branches(loop1_size=6, bridge_size=3, loop2_size=6):
    """Dumbbell with branches from loop, bridge, and other loop."""
    G = make_dumbbell(loop1_size, bridge_size, loop2_size)
    num = G.number_of_nodes()

    # Branch from loop1 (node 3)
    G.add_edge(3, num)
    G.add_edge(num, num + 1)
    G.add_edge(num + 1, num + 2)

    # Branch from bridge (node 7)
    G.add_edge(7, num + 3)
    G.add_edge(num + 3, num + 4)

    # Branch from loop2 (node 12)
    G.add_edge(12, num + 5)
    G.add_edge(num + 5, num + 6)
    G.add_edge(num + 6, num + 7)
    return G


def _sim_summary(name, nodes, edges):
    print(f"{name}: {nodes.position.shape[0]} nodes, "
          f"{edges.pairs.shape[0]} edges, "
          f"{edges.bending_pairs.shape[0]} bending pairs")


class TestDumbbellSimulation:
    """Run a ~80-node dumbbell through the full simulation pipeline."""

    def test_dumbbell_simulation(self):
        G = make_dumbbell(loop1_size=30, bridge_size=20, loop2_size=30)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("Dumbbell", nodes, edges)
        run_and_save(nodes, edges, fields, "dumbbell_simulation.gif")


class TestFlowerSimulation:
    """Flower: hub with 3 petal rings."""

    def test_flower_simulation(self):
        G = make_flower(petal_sizes=(18, 10, 18))
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("Flower", nodes, edges)
        run_and_save(nodes, edges, fields, "flower_simulation.gif")

class TestFlowerSimulation:
    """Flower: hub with 4 petal rings."""

    def test_3_petal_flower_simulation(self):
        G = make_flower(petal_sizes=(18, 14, 18, 14))
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("Flower", nodes, edges)
        run_and_save(nodes, edges, fields, "flower_simulation_4_petal.gif")

    def test_4_petal_flower_simulation(self):
        G = make_chain_of_rings(ring_sizes=(18, 14, 18))
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("ChainOfRings", nodes, edges)
        run_and_save(nodes, edges, fields, "chain_of_rings_simulation.gif")

    def test_flower_long_bridges_simulation(self):
        G = make_flower(petal_sizes=(18, 15, 9), bridge_length=12)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("FlowerLongBridges", nodes, edges)
        run_and_save(nodes, edges, fields, "flower_long_bridges_simulation.gif")

class TestBowtieSimulation:
    """Bowtie: two triangles sharing a cut vertex."""

    def test_bowtie_simulation(self):
        G = make_bowtie()
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("Bowtie", nodes, edges)
        run_and_save(nodes, edges, fields, "bowtie_simulation.gif")


class TestDumbbellWithBranchesSimulation:
    """Dumbbell with branches from loop, bridge, and other loop."""

    def test_dumbbell_with_branches_simulation(self):
        G = make_dumbbell_with_branches(loop1_size=30, bridge_size=10, loop2_size=20)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("DumbbellBranches", nodes, edges)
        run_and_save(nodes, edges, fields, "dumbbell_branches_simulation.gif")


# ---------------------------------------------------------------------------
# Additional topology factories
# ---------------------------------------------------------------------------

def make_lollipop(chain_length=20, loop_size=20):
    """Single chain ending in a loop (like a lollipop)."""
    G = nx.Graph()
    for i in range(chain_length - 1):
        G.add_edge(i, i + 1)
    loop_start = chain_length - 1
    for i in range(loop_size):
        G.add_edge(loop_start + i, loop_start + (i + 1) % loop_size)
    return G


def make_double_lollipop(chain_length=20, loop1_size=15, loop2_size=25):
    """Loop -- chain -- loop: two loops connected by a bridge chain."""
    G = nx.Graph()
    # Left loop: nodes 0..loop1_size-1
    for i in range(loop1_size):
        G.add_edge(i, (i + 1) % loop1_size)
    # Bridge chain from node 0 (entry of left loop)
    bridge_start = loop1_size
    prev = 0
    for i in range(chain_length):
        G.add_edge(prev, bridge_start + i)
        prev = bridge_start + i
    # Right loop: starts at last bridge node
    r_start = bridge_start + chain_length - 1
    for i in range(loop2_size):
        G.add_edge(r_start + i, r_start + (i + 1) % loop2_size)
    return G


def make_caterpillar(spine_length=20, num_teeth=8, tooth_length=5):
    """Spine (line) with evenly spaced branch chains (teeth)."""
    G = nx.Graph()
    for i in range(spine_length - 1):
        G.add_edge(i, i + 1)
    node_id = spine_length
    step = max(1, spine_length // (num_teeth + 1))
    for k in range(num_teeth):
        attach = step * (k + 1)
        if attach >= spine_length:
            break
        prev = attach
        for _ in range(tooth_length):
            G.add_edge(prev, node_id)
            prev = node_id
            node_id += 1
    return G


def make_sunflower(num_petals=5, petal_size=10, bridge_length=6):
    """Hub node with arms, each arm ending in a loop."""
    G = nx.Graph()
    hub = 0
    node_id = 1
    for _ in range(num_petals):
        # Bridge from hub to loop entry
        prev = hub
        for _ in range(bridge_length):
            G.add_edge(prev, node_id)
            prev = node_id
            node_id += 1
        # Loop at arm tip
        loop_start = node_id
        for j in range(petal_size):
            G.add_edge(loop_start + j, loop_start + (j + 1) % petal_size)
        # Connect bridge end to loop
        G.add_edge(prev, loop_start)
        node_id += petal_size
    return G


def make_multi_bowtie(num_cycles=4, cycle_size=6):
    """Multiple simple cycles all sharing one central cut vertex (node 0)."""
    G = nx.Graph()
    node_id = 1
    for _ in range(num_cycles):
        # Each cycle: hub(0) -- c0 -- c1 -- ... -- c_{size-2} -- hub(0)
        cycle_nodes = list(range(node_id, node_id + cycle_size - 1))
        G.add_edge(0, cycle_nodes[0])
        for j in range(len(cycle_nodes) - 1):
            G.add_edge(cycle_nodes[j], cycle_nodes[j + 1])
        G.add_edge(cycle_nodes[-1], 0)
        node_id += cycle_size - 1
    return G


def make_necklace(num_beads=5, bead_size=10, connector_length=4):
    """Sequence of loops (beads) connected midpoint-to-midpoint by bridge chains."""
    G = nx.Graph()
    node_id = 0
    bead_mids = []
    for _ in range(num_beads):
        bead_start = node_id
        for j in range(bead_size):
            G.add_edge(bead_start + j, bead_start + (j + 1) % bead_size)
        bead_mids.append(bead_start + bead_size // 2)
        node_id += bead_size
    # Connect consecutive beads with bridge chains between their midpoints
    for i in range(num_beads - 1):
        prev = bead_mids[i]
        for _ in range(connector_length):
            G.add_edge(prev, node_id)
            prev = node_id
            node_id += 1
        G.add_edge(prev, bead_mids[i + 1])
    return G


def make_ring_with_tails(ring_size=16, num_tails=5, tail_length=8):
    """Central ring with evenly spaced open chain tails radiating outward."""
    G = nx.Graph()
    for i in range(ring_size):
        G.add_edge(i, (i + 1) % ring_size)
    node_id = ring_size
    step = max(1, ring_size // num_tails)
    for k in range(num_tails):
        attach = (k * step) % ring_size
        prev = attach
        for _ in range(tail_length):
            G.add_edge(prev, node_id)
            prev = node_id
            node_id += 1
    return G


def make_binary_tree_of_loops(depth=2, loop_size=6, bridge_length=4):
    """Binary tree of bridge chains; every leaf is a small loop."""
    G = nx.Graph()
    counter = [0]

    def new_node():
        n = counter[0]
        counter[0] += 1
        return n

    def add_loop(attach):
        """Add a loop_size cycle attached to `attach`."""
        first = new_node()
        G.add_edge(attach, first)
        prev = first
        for _ in range(loop_size - 1):
            n = new_node()
            G.add_edge(prev, n)
            prev = n
        G.add_edge(prev, first)  # close the cycle

    def add_bridge_tip(from_node):
        prev = from_node
        for _ in range(bridge_length):
            n = new_node()
            G.add_edge(prev, n)
            prev = n
        return prev

    def build(attach, d):
        if d == 0:
            add_loop(attach)
            return
        for _ in range(2):
            tip = add_bridge_tip(attach)
            build(tip, d - 1)

    root = new_node()
    build(root, depth)
    return G


def make_spine_of_rings(num_rings=5, ring_size=8, bridge_length=3):
    """Alternating rings and bridge chains along a linear spine."""
    G = nx.Graph()
    node_id = 0

    def add_ring():
        nonlocal node_id
        start = node_id
        for j in range(ring_size):
            G.add_edge(start + j, start + (j + 1) % ring_size)
        node_id += ring_size
        return start, start + ring_size // 2  # entry, exit midpoints

    def add_bridge(from_node):
        nonlocal node_id
        prev = from_node
        for _ in range(bridge_length):
            G.add_edge(prev, node_id)
            prev = node_id
            node_id += 1
        return prev

    entry, exit_mid = add_ring()
    for _ in range(num_rings - 1):
        bridge_end = add_bridge(exit_mid)
        entry, exit_mid = add_ring()
        G.add_edge(bridge_end, entry)

    return G


def make_asymmetric_tree_of_rings(small_ring=5, large_ring=20, bridge=8):
    """Asymmetric: small loop -- bridge -- large loop -- bridge -- small loop."""
    G = nx.Graph()
    node_id = 0

    def add_ring(size, attach=None):
        nonlocal node_id
        start = node_id
        for j in range(size):
            G.add_edge(start + j, start + (j + 1) % size)
        if attach is not None:
            G.add_edge(attach, start)
        node_id += size
        return start + size // 2  # midpoint for bridging

    def add_bridge(from_node, length):
        nonlocal node_id
        prev = from_node
        for _ in range(length):
            G.add_edge(prev, node_id)
            prev = node_id
            node_id += 1
        return prev

    mid1 = add_ring(small_ring)
    b1 = add_bridge(mid1, bridge)
    mid2 = add_ring(large_ring, attach=b1)
    b2 = add_bridge(mid2, bridge)
    add_ring(small_ring, attach=b2)
    return G


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------

def _assert_simulation_state_valid(nodes, edges):
    """Basic sanity checks on nodes/edges after build."""
    n = nodes.position.shape[0]
    e = edges.pairs.shape[0]
    assert n > 0
    assert e > 0
    assert edges.pairs.shape == (e, 2)
    assert edges.theta.shape == (e,)
    assert edges.rest_lengths.shape == (e,)
    assert edges.bending_pairs.ndim == 2 and edges.bending_pairs.shape[1] == 2


# ---------------------------------------------------------------------------
# Validation tests (no simulation — just structural checks)
# ---------------------------------------------------------------------------

class TestValidateRodTopology:
    """validate_rod_topology rejects invalid graphs and accepts valid cactus graphs."""

    def test_rejects_self_loop(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(1, 1)
        with pytest.raises(ValueError, match="self-loop"):
            validate_rod_topology(G)

    def test_rejects_wheel_graph(self):
        """Wheel W6: hub connects to all cycle nodes, breaking cactus property."""
        with pytest.raises(ValueError):
            validate_rod_topology(nx.wheel_graph(6))

    def test_rejects_petersen_graph(self):
        """Petersen graph is 3-regular — definitely not a cactus."""
        with pytest.raises(ValueError):
            validate_rod_topology(nx.petersen_graph())

    def test_rejects_cube_graph(self):
        """Hypercube Q3: 3-regular, multiple shared-edge cycles."""
        with pytest.raises(ValueError):
            validate_rod_topology(nx.hypercube_graph(3))


    def test_accepts_binary_tree_of_loops(self):
        validate_rod_topology(make_binary_tree_of_loops(2, 5, 3))

    def test_accepts_spine_of_rings(self):
        validate_rod_topology(make_spine_of_rings(3, 6, 2))


class TestBuildDirectedGraph:
    """Structural checks on the directed graph produced by build_directed_cosserat_graph."""

    def test_directed_graph_has_same_node_count(self):
        G = make_dumbbell(8, 4, 8)
        dG, _ = build_directed_cosserat_graph(G)
        assert dG.number_of_nodes() == G.number_of_nodes()

    def test_directed_graph_has_same_edge_count(self):
        G = make_dumbbell(8, 4, 8)
        dG, _ = build_directed_cosserat_graph(G)
        assert dG.number_of_edges() == G.number_of_edges()

    def test_root_is_valid_node(self):
        G = make_lollipop(10, 8)
        dG, root = build_directed_cosserat_graph(G)
        assert root in dG.nodes()

    def test_directed_graph_is_weakly_connected(self):
        G = make_necklace(3, 6, 2)
        dG, _ = build_directed_cosserat_graph(G)
        assert nx.is_weakly_connected(dG)

    def test_no_isolated_nodes(self):
        G = make_caterpillar(10, 4, 3)
        dG, _ = build_directed_cosserat_graph(G)
        for n in dG.nodes():
            assert dG.in_degree(n) + dG.out_degree(n) > 0, \
                f"Node {n} is isolated in directed graph"

    def test_lollipop_loop_edges_form_cycle(self):
        """After directing, the loop edges should still form a cycle in the directed graph."""
        G = make_lollipop(5, 6)
        dG, _ = build_directed_cosserat_graph(G)
        cycles = list(nx.simple_cycles(dG))
        assert len(cycles) == 1, f"Expected 1 cycle, got {len(cycles)}"

    def test_dumbbell_has_two_directed_cycles(self):
        G = make_dumbbell(6, 3, 6)
        dG, _ = build_directed_cosserat_graph(G)
        cycles = list(nx.simple_cycles(dG))
        assert len(cycles) == 2, f"Expected 2 cycles, got {len(cycles)}"


    def test_simulation_state_valid_lollipop(self):
        G = make_lollipop(10, 8)
        nodes, edges, _ = build_simulation_from_undirected(G)
        _assert_simulation_state_valid(nodes, edges)

    def test_simulation_state_valid_necklace(self):
        G = make_necklace(3, 6, 2)
        nodes, edges, _ = build_simulation_from_undirected(G)
        _assert_simulation_state_valid(nodes, edges)

    def test_simulation_state_valid_binary_tree_of_loops(self):
        G = make_binary_tree_of_loops(2, 5, 3)
        nodes, edges, _ = build_simulation_from_undirected(G)
        _assert_simulation_state_valid(nodes, edges)

    def test_simulation_state_valid_multi_bowtie(self):
        G = make_multi_bowtie(3, 6)
        nodes, edges, _ = build_simulation_from_undirected(G)
        _assert_simulation_state_valid(nodes, edges)

    def test_simulation_state_valid_ring_with_tails(self):
        G = make_ring_with_tails(10, 4, 5)
        nodes, edges, _ = build_simulation_from_undirected(G)
        _assert_simulation_state_valid(nodes, edges)


# ---------------------------------------------------------------------------
# New visual simulation tests
# ---------------------------------------------------------------------------

class TestLollipopSimulation:
    """Chain ending in a single loop."""
    def test_big_loop_lollipop(self):
        """Small chain, large loop."""
        G = make_lollipop(chain_length=8, loop_size=40)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("LollipopBigLoop", nodes, edges)
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "lollipop_big_loop_simulation.gif")

class TestCaterpillarSimulation:
    """Spine with many branch chains."""

    def test_caterpillar_simulation(self):
        G = make_caterpillar(spine_length=20, num_teeth=6, tooth_length=8)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("Caterpillar(6 teeth)", nodes, edges)
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "caterpillar_simulation.gif")

class TestSunflowerSimulation:
    """Hub with arms ending in loops."""

    def test_sunflower_simulation(self):
        G = make_sunflower(num_petals=5, petal_size=10, bridge_length=6)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("Sunflower(5 petals)", nodes, edges)
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "sunflower_simulation.gif")

    def test_sunflower_many_petals(self):
        G = make_sunflower(num_petals=8, petal_size=8, bridge_length=4)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("Sunflower(8 petals)", nodes, edges)
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "sunflower_many_petals_simulation.gif")


class TestMultiBowtieSimulation:
    """Multiple simple cycles sharing one cut vertex."""

    def test_triple_bowtie(self):
        """Three triangles at one node."""
        G = make_multi_bowtie(num_cycles=3, cycle_size=5)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("MultiBowtie(3)", nodes, edges)
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "multi_bowtie_3_simulation.gif")

    def test_5_cycle_bowtie(self):
        """Five larger cycles at one node."""
        G = make_multi_bowtie(num_cycles=5, cycle_size=8)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("MultiBowtie(5 × 8)", nodes, edges)
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "multi_bowtie_5_simulation.gif")


class TestNecklaceSimulation:
    """Loops connected in sequence by bridge chains."""

    def test_necklace_many_beads(self):
        G = make_necklace(num_beads=8, bead_size=7, connector_length=3)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("Necklace(8 beads)", nodes, edges)
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "necklace_many_beads_simulation.gif")


class TestRingWithTailsSimulation:
    """Central ring with open-chain tails."""
    def test_ring_with_many_short_tails(self):
        G = make_ring_with_tails(ring_size=20, num_tails=8, tail_length=4)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("RingWithTails(8 short)", nodes, edges)
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "ring_with_tails_many_simulation.gif")


class TestBinaryTreeOfLoopsSimulation:
    """Binary branching tree where leaves are loops."""
    def test_binary_tree_depth3(self):
        """8 leaf loops — larger and more complex."""
        G = make_binary_tree_of_loops(depth=3, loop_size=5, bridge_length=3)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("BinaryTreeOfLoops(depth=3)", nodes, edges)
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "binary_tree_loops_depth3.gif")


class TestSpineOfRingsSimulation:
    """Rings connected along a linear spine."""
    def test_spine_of_rings_long_bridges(self):
        G = make_spine_of_rings(num_rings=4, ring_size=10, bridge_length=8)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("SpineOfRings(long bridges)", nodes, edges)
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "spine_of_rings_long_bridges.gif")


class TestAsymmetricTreeOfRingsSimulation:
    """Small--bridge--large--bridge--small ring topology."""

    def test_asymmetric_tree_of_rings(self):
        G = make_asymmetric_tree_of_rings(small_ring=6, large_ring=22, bridge=8)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("AsymmetricTreeOfRings", nodes, edges)
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "asymmetric_tree_of_rings.gif")


class TestLargeScaleSimulation:
    """Stress tests with 150+ node graphs."""
    def test_large_binary_tree_of_loops(self):
        G = make_binary_tree_of_loops(depth=3, loop_size=6, bridge_length=5)
        nodes, edges, fields = build_simulation_from_undirected(G)
        _sim_summary("LargeBinaryTreeOfLoops", nodes, edges)
        assert nodes.position.shape[0] > 100
        _assert_simulation_state_valid(nodes, edges)
        run_and_save(nodes, edges, fields, "large_binary_tree_loops.gif",
                     num_steps=300)
