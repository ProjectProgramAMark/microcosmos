import jax
import jax.numpy as jnp
import numpy as np

import networkx as nx

from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.structs.fields import Fields
from microcosmos.utils import displacement
from microcosmos.cactus_graph_layout import layout_positions_cactus

def validate_rod_topology(G):
    """
    Validate that an undirected graph is a valid rod topology for Cosserat
    directed-graph conversion.

    A valid rod topology is a Cactus graph: 
    a graph whose every biconnected component is
    either a single edge (bridge / chain segment) or a simple cycle (loop).
    This guarantees that bridge removal isolates clean rings where
    nx.find_cycle can trace every edge.

    Invalid topologies include:
    - Self-loops
    - Parallel edges (multigraph)
    - Complete subgraphs (K4, K5, etc.) where cycles share edges
    - Multiple cycles sharing a node without bridge separation
    - Any biconnected component with a node of degree > 2
      (e.g. theta graphs, wheel graphs, barbell_graph with complete ends)

    Args:
        G: An undirected networkx.Graph

    Raises:
        TypeError:  If G is not an undirected Graph.
        ValueError: If the topology violates rod constraints, with a
                    message describing the specific problem.
    """
    if not isinstance(G, nx.Graph) or isinstance(G, nx.DiGraph):
        raise TypeError(
            f"Expected an undirected networkx.Graph, got {type(G).__name__}"
        )

    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty (no nodes)")

    if G.number_of_edges() == 0:
        raise ValueError("Graph has no edges")

    # Self-loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        raise ValueError(
            f"Graph contains self-loops at nodes: "
            f"{[u for u, _ in self_loops]}"
        )

    # Multigraph check (shouldn't happen with nx.Graph, but guard against
    # someone passing a converted MultiGraph)
    if isinstance(G, nx.MultiGraph):
        raise TypeError(
            "MultiGraph not supported — each bond must be a single edge"
        )

    # Check biconnected components: each must be a single edge or simple cycle
    for comp_nodes in nx.biconnected_components(G):
        subgraph = G.subgraph(comp_nodes)

        if subgraph.number_of_edges() == 1:
            continue  # single edge (bridge) — always valid

        # Multi-edge component: every node must have degree exactly 2
        # (i.e. it must be a simple cycle)
        bad_nodes = [
            (n, d) for n, d in subgraph.degree() if d != 2
        ]
        if bad_nodes:
            node_str = ", ".join(
                f"node {n} (degree {d})" for n, d in bad_nodes[:5]
            )
            raise ValueError(
                f"Biconnected component {sorted(comp_nodes)} is not a "
                f"simple cycle — contains nodes with degree != 2: "
                f"{node_str}. This typically means multiple cycles share "
                f"nodes/edges without bridge separation (e.g. complete "
                f"subgraph, theta graph). Use simple cycles connected by "
                f"bridge chains instead."
            )


def build_directed_cosserat_graph(G):
    """
    Converts an arbitrary undirected rod topology into a Directed Acyclic-ish Graph
    optimized for Cosserat bending pairs, avoiding sinks in simple loops.
    """
    validate_rod_topology(G)

    # 1. Handle disconnected graphs (process the largest physical structure)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
    # 2. Find the optimal Macro-Root
    eccentricities = nx.eccentricity(G)
    leaves = [n for n, d in G.degree() if d == 1]
    
    if leaves:
        # Branched structure: pick the furthest leaf to maximize trunk length
        root_node = max(leaves, key=lambda n: eccentricities[n])
    else:
        # Closed loops (like dumbbells): pick an extreme edge node
        root_node = nx.periphery(G)[0]

    # Initialize the new directed graph and copy node data
    directed_G = nx.DiGraph()
    directed_G.add_nodes_from(G.nodes(data=True))
    
    # 3. Calculate macro-flow distances from the root
    distances = nx.shortest_path_length(G, source=root_node)
    
    # 4. Segment the graph: Extract Bridges (Chains)
    bridges = list(nx.bridges(G))
    
    # Orient the bridges down the gradient (away from root)
    for u, v in bridges:
        if distances[u] < distances[v]:
            directed_G.add_edge(u, v, **G.edges[u, v])
        else:
            directed_G.add_edge(v, u, **G.edges[u, v])
            
    # 5. Segment the graph: Extract Blocks (Loops)
    # Use biconnected components to find individual cycles.
    # This handles cases like the bowtie where two cycles share a cut vertex
    # but have no bridges between them — connected_components would merge them
    # into one component, but biconnected_components correctly separates them.
    loop_G = G.copy()
    loop_G.remove_edges_from(bridges)

    for comp_nodes in nx.biconnected_components(loop_G):
        if len(comp_nodes) < 3:
            continue  # Skip single edges or isolated nodes

        loop_subgraph = loop_G.subgraph(comp_nodes)
        entry_node = min(comp_nodes, key=lambda n: distances[n])

        try:
            cycle_edges = nx.find_cycle(loop_subgraph, source=entry_node)
            for u, v in cycle_edges:
                directed_G.add_edge(u, v, **G.edges[u, v])
        except nx.NetworkXNoCycle:
            pass
            
    return directed_G, root_node

def make_edges(num_nodes: int, graph: str = "loop", num_instances: int = 1) -> tuple[jax.Array, jax.Array]:
    """
    Create a directed edge list for graph topologies.

    Each undirected bond is stored as ONE directed edge (forward along chain,
    parent->child for branches).

    Args:
        num_nodes: Total number of nodes across all instances
        graph: Topology type ('loop', 'line', or 'tree')
        num_instances: Number of independent instances

    Returns:
        edges: (E, 2) int32 array of (source, target) pairs
        edge_departure_angles: (E,) float32 structural angle offsets
    """
    nodes_per_instance = num_nodes // num_instances
    remainder = num_nodes % num_instances

    if nodes_per_instance < 1:
        raise ValueError(f"num_nodes ({num_nodes}) must be at least {num_instances} to create {num_instances} instances.")

    all_edges = []
    all_departure_angles = []

    start = 0
    for instance_idx in range(num_instances):
        size = nodes_per_instance + (1 if instance_idx < remainder else 0)
        inst_edges, inst_departures = _make_single_edges(size, graph, start)
        all_edges.append(inst_edges)
        all_departure_angles.append(inst_departures)
        start += size

    edges = np.concatenate(all_edges, axis=0)
    departure_angles = np.concatenate(all_departure_angles, axis=0)

    return jnp.array(edges, dtype=jnp.int32), jnp.array(departure_angles, dtype=jnp.float32)


def _make_single_edges(num_nodes: int, graph: str, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Create edges for a single instance.

    Returns numpy arrays:
        edges: (E, 2) int array of (source, target) pairs
        departure_angles: (E,) float array of departure angles
    """
    if graph == "loop":
        # E = N edges: each node connects to the next, wrapping around
        sources = np.arange(num_nodes) + offset
        targets = np.roll(sources, -1)
        edges = np.stack([sources, targets], axis=1)
        departure_angles = np.zeros(num_nodes, dtype=np.float32)

    elif graph == "line":
        # E = N-1 edges: each node connects to the next (no wrap)
        sources = np.arange(num_nodes - 1) + offset
        targets = sources + 1
        edges = np.stack([sources, targets], axis=1)
        departure_angles = np.zeros(num_nodes - 1, dtype=np.float32)

    elif graph == "tree":
        # Trunk + two branches
        TRUNK_SIZE = int(num_nodes * 0.6)
        REMAINING = num_nodes - TRUNK_SIZE
        BRANCH_1_SIZE = REMAINING // 2
        BRANCH_2_SIZE = REMAINING - BRANCH_1_SIZE

        TRUNK_START = offset
        BRANCH_1_START = offset + TRUNK_SIZE
        BRANCH_2_START = BRANCH_1_START + BRANCH_1_SIZE
        JUNCTION_IDX = TRUNK_START + TRUNK_SIZE // 2

        edge_list = []
        dep_list = []

        # Trunk edges: (i, i+1) for i in 0..TRUNK_SIZE-2
        for i in range(TRUNK_SIZE - 1):
            edge_list.append((TRUNK_START + i, TRUNK_START + i + 1))
            dep_list.append(0.0)

        # Branch 1 edges: junction -> branch_start, then chain
        edge_list.append((JUNCTION_IDX, BRANCH_1_START))
        dep_list.append(np.pi / 2.0)  # Right branch departure

        for i in range(BRANCH_1_SIZE - 1):
            edge_list.append((BRANCH_1_START + i, BRANCH_1_START + i + 1))
            dep_list.append(np.pi / 2.0)  # Continue in branch direction

        # Branch 2 edges: junction -> branch_start, then chain
        edge_list.append((JUNCTION_IDX, BRANCH_2_START))
        dep_list.append(-np.pi / 2.0)  # Left branch departure

        for i in range(BRANCH_2_SIZE - 1):
            edge_list.append((BRANCH_2_START + i, BRANCH_2_START + i + 1))
            dep_list.append(-np.pi / 2.0)  # Continue in branch direction

        edges = np.array(edge_list, dtype=np.int32)
        departure_angles = np.array(dep_list, dtype=np.float32)

    else:
        raise ValueError(f"Unknown graph type: {graph}")

    return edges, departure_angles



def initialize_positions(num_nodes: int, graph_type: str, shape: jax.Array, key: jax.Array, num_instances: int = 1) -> jax.Array:
    """ Initialize node positions for different graph types. For 'tree', place two branches on opposite sides. """
    nodes_per_instance = num_nodes // num_instances
    remainder = num_nodes % num_instances

    key_centers, key_positions = jax.random.split(key)

    if num_instances == 1:
        centers = (shape / 2).reshape(1, 2)
    else:
        centers = jax.random.uniform(
            key_centers,
            (num_instances, 2),
            minval=0.3 * shape,
            maxval=0.7 * shape
        )

    positions_list = []
    keys = jax.random.split(key_positions, num_instances)

    for i in range(num_instances):
        size = nodes_per_instance + (1 if i < remainder else 0)

        if graph_type == "tree":
            # --- Custom tree initialization: vertical trunk + two horizontal branches ---
            # Must match the splits in _make_single_edges
            TRUNK_SIZE = int(size * 0.6)
            REMAINING = size - TRUNK_SIZE
            BRANCH1_SIZE = REMAINING // 2
            BRANCH2_SIZE = REMAINING - BRANCH1_SIZE

            # Trunk: straight line along y-axis (vertical)
            trunk_start = centers[i] - jnp.array([0.0, 0.4 * min(shape)])
            trunk_end = centers[i] + jnp.array([0.0, 0.4 * min(shape)])
            trunk_positions = jnp.linspace(trunk_start, trunk_end, TRUNK_SIZE)

            # Branch point (middle of trunk)
            branch_point = trunk_positions[TRUNK_SIZE // 2]

            # Branch 1: right (0 degrees)
            branch1_dir = jnp.array([1.0, 0.0])
            branch1_length = 0.3 * min(shape)
            branch1_steps = jnp.linspace(1, BRANCH1_SIZE, BRANCH1_SIZE) / (BRANCH1_SIZE + 1)
            branch1_positions = branch_point + branch1_steps[:, None] * branch1_dir * branch1_length

            # Branch 2: left (180 degrees)
            branch2_dir = jnp.array([-1.0, 0.0])
            branch2_length = 0.3 * min(shape)
            branch2_steps = jnp.linspace(1, BRANCH2_SIZE, BRANCH2_SIZE) / (BRANCH2_SIZE + 1)
            branch2_positions = branch_point + branch2_steps[:, None] * branch2_dir * branch2_length

            # Add small noise to all positions
            all_positions = jnp.concatenate([trunk_positions, branch1_positions, branch2_positions], axis=0)
            noise = jax.random.normal(keys[i], all_positions.shape) * (0.02 * min(shape))
            instance_positions = all_positions + noise

        elif graph_type == "line":
            # Vertical line initialization
            line_length = min(shape) * 0.8
            line_start = centers[i] - jnp.array([0.0, line_length / 2])
            line_end = centers[i] + jnp.array([0.0, line_length / 2])
            line_positions = jnp.linspace(line_start, line_end, size)

            # Add small noise
            # noise = jax.random.normal(keys[i], line_positions.shape) * (0.02 * min(shape))
            instance_positions = line_positions

        else:
            # Default (loop): put in noisy circle
            radius = min(shape) * 0.25
            noise_scale = radius * 0.05
            angles = jnp.linspace(0, 2 * jnp.pi, size, endpoint=False)
            circle_positions = jnp.stack([
                radius * jnp.cos(angles),
                radius * jnp.sin(angles)
            ], axis=1)
            noise = jax.random.normal(keys[i], circle_positions.shape) * noise_scale
            instance_positions = centers[i] + circle_positions + noise

        positions_list.append(instance_positions)

    return jnp.concatenate(positions_list, axis=0)


def compute_bending_pairs(edge_pairs, departure_angles, num_nodes):
    """
    Compute bending constraint pairs from edge topology.

    For each node that is both a target of some edge (incoming) and a source
    of some edge (outgoing), create bending pairs between each incoming edge
    and each outgoing edge. This follows Cosserat rod semantics where curvature
    is measured along the material curve direction.

    At chain interior nodes (1 in, 1 out): 1 bending pair
    At junctions (1 in, 2 out for degree-3): 2 bending pairs
    At chain endpoints / branch tips: 0 bending pairs (no constraint)

    Args:
        edge_pairs: (E, 2) int32 array of (src, tgt) pairs
        departure_angles: (E,) float32 array of departure angles (intermediate, not stored)
        num_nodes: Total number of nodes

    Returns:
        bending_pairs: (B, 2) int32 — (edge_in_idx, edge_out_idx)
        bending_rest_angles: (B,) float32 — initial rest angles from departure angle differences
    """
    edges_np = np.array(edge_pairs)
    dep_np = np.array(departure_angles)
    num_edges = edges_np.shape[0]

    # Build node -> incoming edges and node -> outgoing edges
    incoming = [[] for _ in range(num_nodes)]  # edges whose tgt is this node
    outgoing = [[] for _ in range(num_nodes)]  # edges whose src is this node

    for e_idx in range(num_edges):
        src, tgt = int(edges_np[e_idx, 0]), int(edges_np[e_idx, 1])
        outgoing[src].append(e_idx)
        incoming[tgt].append(e_idx)

    bp_list = []
    rest_list = []

    for node in range(num_nodes):
        for e_in in incoming[node]:
            for e_out in outgoing[node]:
                bp_list.append((e_in, e_out))
                # Rest angle = departure_angle[e_out] - departure_angle[e_in]
                # For chain edges (both 0.0) -> rest_angle = 0
                # For junction->branch (dep=±pi/2) -> rest_angle = ±pi/2
                rest_list.append(float(dep_np[e_out] - dep_np[e_in]))

    if len(bp_list) == 0:
        bending_pairs = np.zeros((0, 2), dtype=np.int32)
        bending_rest_angles = np.zeros(0, dtype=np.float32)
    else:
        bending_pairs = np.array(bp_list, dtype=np.int32)
        bending_rest_angles = np.array(rest_list, dtype=np.float32)

    return jnp.array(bending_pairs, dtype=jnp.int32), jnp.array(bending_rest_angles, dtype=jnp.float32)


def set_orthogonal_rest_angles(edge_pairs, edge_theta, bending_pairs,
                               bending_rest_angles, num_nodes):
    """
    Post-process bending rest angles to enforce orthogonality at junction nodes.

    At each junction node (in-degree + out-degree > 2), identifies which bending
    pair is the "continuation" (most aligned edges) and which are "branches".
    Continuation pairs keep rest_angle = 0; branch pairs get rest_angle = ±π/2
    with sign determined by layout geometry.

    Chain nodes (1 in, 1 out) are not modified — their rest angles stay at 0.

    Args:
        edge_pairs: (E, 2) int32 array of (src, tgt) pairs
        edge_theta: (E,) float32 array of per-edge orientation from layout
        bending_pairs: (B, 2) int32 array of (edge_in, edge_out) pairs
        bending_rest_angles: (B,) float32 array of current rest angles
        num_nodes: Total number of nodes

    Returns:
        bending_rest_angles: (B,) float32 array with junction angles set to ±π/2
    """
    edges_np = np.array(edge_pairs)
    theta_np = np.array(edge_theta)
    bp_np = np.array(bending_pairs)
    rest_np = np.array(bending_rest_angles)

    if bp_np.shape[0] == 0:
        return bending_rest_angles

    # Build incidence lists
    incoming = [[] for _ in range(num_nodes)]
    outgoing = [[] for _ in range(num_nodes)]
    for e_idx in range(edges_np.shape[0]):
        src, tgt = int(edges_np[e_idx, 0]), int(edges_np[e_idx, 1])
        outgoing[src].append(e_idx)
        incoming[tgt].append(e_idx)

    # Find junction nodes
    junction_nodes = set()
    for node in range(num_nodes):
        if len(incoming[node]) + len(outgoing[node]) > 2:
            junction_nodes.add(node)

    if not junction_nodes:
        return bending_rest_angles

    # Group bending pairs by their shared node
    # Shared node = tgt of e_in = src of e_out
    for bp_idx in range(bp_np.shape[0]):
        e_in, e_out = int(bp_np[bp_idx, 0]), int(bp_np[bp_idx, 1])
        shared = int(edges_np[e_in, 1])

        if shared not in junction_nodes:
            continue

        # Compute the actual angular difference between in and out edges
        theta_in = float(theta_np[e_in])
        theta_out = float(theta_np[e_out])
        actual_diff = float(np.arctan2(
            np.sin(theta_out - theta_in),
            np.cos(theta_out - theta_in),
        ))

        # Is this the continuation pair (most aligned) or a branch pair?
        # At this shared node, find the bending pair with smallest |actual_diff|
        # That one is the continuation; all others are branches.
        # We need to check all bending pairs at this node to decide.
        pass  # Handled below

    # Process each junction node: find continuation pair, set branch pairs to ±π/2
    for node in junction_nodes:
        in_edges = incoming[node]
        out_edges = outgoing[node]

        # Collect all bending pair indices at this node
        node_bp_indices = []
        for bp_idx in range(bp_np.shape[0]):
            e_in = int(bp_np[bp_idx, 0])
            if int(edges_np[e_in, 1]) == node:
                node_bp_indices.append(bp_idx)

        if not node_bp_indices:
            continue

        # Compute |actual_diff| for each bending pair at this node
        abs_diffs = []
        for bp_idx in node_bp_indices:
            e_in, e_out = int(bp_np[bp_idx, 0]), int(bp_np[bp_idx, 1])
            theta_in = float(theta_np[e_in])
            theta_out = float(theta_np[e_out])
            diff = float(np.arctan2(
                np.sin(theta_out - theta_in),
                np.cos(theta_out - theta_in),
            ))
            abs_diffs.append(abs(diff))

        # The continuation pair has the smallest angular difference
        continuation_local_idx = int(np.argmin(abs_diffs))

        for local_idx, bp_idx in enumerate(node_bp_indices):
            if local_idx == continuation_local_idx:
                rest_np[bp_idx] = 0.0  # continuation: straight
            else:
                # Branch: ±π/2, sign from layout geometry
                e_in, e_out = int(bp_np[bp_idx, 0]), int(bp_np[bp_idx, 1])
                theta_in = float(theta_np[e_in])
                theta_out = float(theta_np[e_out])
                actual_diff = float(np.arctan2(
                    np.sin(theta_out - theta_in),
                    np.cos(theta_out - theta_in),
                ))
                sign = 1.0 if actual_diff >= 0 else -1.0
                rest_np[bp_idx] = sign * (np.pi / 2)

    return jnp.array(rest_np, dtype=jnp.float32)



# ---------------------------------------------------------------------------
# LBM helpers (D2Q9)
# ---------------------------------------------------------------------------
_LBM_VELOCITIES = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
])
_LBM_WEIGHTS = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])


def _initialize_f_from_velocity(velocity_field, density=1.0):
    u = jnp.moveaxis(velocity_field, 0, -1)
    u_dot_v = jnp.einsum("xyc, ic -> ixy", u, _LBM_VELOCITIES)
    u_sq = jnp.sum(u**2, axis=-1)
    feq = density * _LBM_WEIGHTS[:, None, None] * (
        1.0 + 3.0 * u_dot_v + 4.5 * (u_dot_v**2) - 1.5 * u_sq[None, :, :]
    )
    return feq



# ---------------------------------------------------------------------------
# Simulation builders from NetworkX graphs
# ---------------------------------------------------------------------------

def relabel_nodes_bfs(directed_G, root):
    """Relabel nodes to contiguous 0..N-1 in BFS order from root."""
    bfs_order = list(nx.bfs_tree(directed_G, root).nodes())
    remaining = [n for n in directed_G.nodes() if n not in bfs_order]
    bfs_order.extend(remaining)
    label_map = {old: new for new, old in enumerate(bfs_order)}
    return nx.relabel_nodes(directed_G, label_map), label_map[root]


def layout_positions(directed_G, edge_list, size, rest_length, method="cactus", root=0):
    """Layout node positions using the specified method, centered in domain.

    Args:
        method: "cactus" (structure-aware, default) or "kamada_kawai".
    """
    if method == "kamada_kawai":
        num_nodes = directed_G.number_of_nodes()
        pos_nx = nx.kamada_kawai_layout(directed_G)
        positions = np.array([pos_nx[i] for i in range(num_nodes)])

        edge_arr = np.array(edge_list)
        edge_vectors = positions[edge_arr[:, 1]] - positions[edge_arr[:, 0]]
        mean_edge_len = np.mean(np.linalg.norm(edge_vectors, axis=1))
        scale = rest_length / (mean_edge_len + 1e-8)
        positions = positions * scale

    elif method == "cactus":
        undirected_G = directed_G.to_undirected()
        positions = layout_positions_cactus(undirected_G, root, rest_length)

    else:
        raise ValueError(f"Unknown layout method: {method!r}")

    center = np.array(size) / 2.0
    positions = positions - positions.mean(axis=0) + center
    return jnp.array(positions, dtype=jnp.float32)


def make_fields(size):
    """Create zeroed-out Fields for a given domain size."""
    h, w = size
    fluid_velocity = jnp.zeros((2, h, w))
    f_grid = _initialize_f_from_velocity(fluid_velocity)
    return Fields(
        grid_shape=(h, w),
        steric=jnp.zeros((h, w)),
        fluid_velocity=fluid_velocity,
        f_grid=f_grid,
    )


def build_simulation_from_directed(directed_G, root, shape=(128, 128), rest_length=2.0, layout="cactus"):
    """
    Take a directed NetworkX graph (already oriented for Cosserat), lay out
    positions, and return (nodes, edges, fields) ready for simulate().

    Args:
        layout: "cactus" (structure-aware, default) or "kamada_kawai".
    """
    size = tuple(int(s) for s in shape)
    directed_G, root = relabel_nodes_bfs(directed_G, root)

    edge_list = list(directed_G.edges())
    edge_pairs = jnp.array(edge_list, dtype=jnp.int32)
    num_nodes = directed_G.number_of_nodes()
    num_edges = edge_pairs.shape[0]

    positions = layout_positions(directed_G, edge_list, size, rest_length,
                                 method=layout, root=root)

    # Per-edge theta from laid-out positions
    deltas = displacement(size, positions[edge_pairs[:, 1]], positions[edge_pairs[:, 0]])
    edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

    # Bending pairs — start with all-zero departure angles (straight chains),
    # then post-process to set ±π/2 rest angles at junction nodes
    departure_angles = jnp.zeros(num_edges, dtype=jnp.float32)
    bending_pairs, bending_rest_angles = compute_bending_pairs(
        edge_pairs, departure_angles, num_nodes)
    bending_rest_angles = set_orthogonal_rest_angles(
        edge_pairs, edge_theta, bending_pairs, bending_rest_angles, num_nodes)

    nodes = Nodes(
        position=positions,
        velocity=jnp.zeros_like(positions),
        color_bend=jnp.arange(num_nodes) % 3,
        debug_vector=jnp.zeros_like(positions),
    )
    edges = Edges(
        pairs=edge_pairs,
        theta=edge_theta,
        rest_lengths=jnp.full(num_edges, rest_length),
        bending_pairs=bending_pairs,
        bending_rest_angles=bending_rest_angles,
        bending_stiffness=jnp.ones(bending_pairs.shape[0]) * 0.5,
    )

    return nodes, edges, make_fields(size)


def build_simulation_from_undirected(G_undirected, shape=(128, 128), rest_length=2.0, layout="cactus"):
    """
    Take an undirected NetworkX graph, convert it to a directed Cosserat
    graph, lay out positions, and return (nodes, edges, fields) ready for
    simulate().

    Args:
        layout: "cactus" (structure-aware, default) or "kamada_kawai".
    """
    directed_G, root = build_directed_cosserat_graph(G_undirected)
    return build_simulation_from_directed(directed_G, root, shape, rest_length,
                                          layout=layout)
