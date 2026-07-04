import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Structure-aware cactus layout
# note, this is quite specific to the layouts we support now. 
# Ideally we get rid of this when we leave the arbitrary graph experiments behind.
# ---------------------------------------------------------------------------

def _build_block_cut_tree(G, root):
    """Build a block-cut tree from an undirected cactus graph.

    Returns (bct, root_block, block_data) where bct is a NetworkX tree with
    block nodes ("block", i) and cut-vertex nodes ("cut", v), and block_data
    maps each block id to {"nodes": set, "type": "cycle"|"bridge"}.
    """
    bicomps = list(nx.biconnected_components(G))
    cuts = set(nx.articulation_points(G))

    bct = nx.Graph()
    block_data = {}

    for i, comp_nodes in enumerate(bicomps):
        block_id = ("block", i)
        sg = G.subgraph(comp_nodes)
        is_cycle = sg.number_of_edges() == len(comp_nodes)
        block_data[block_id] = {
            "nodes": set(comp_nodes),
            "type": "cycle" if is_cycle else "bridge",
        }
        bct.add_node(block_id)

        for node in comp_nodes:
            if node in cuts:
                cut_id = ("cut", node)
                bct.add_node(cut_id)
                bct.add_edge(block_id, cut_id)

    # Find root block (the block containing root)
    root_block = None
    for block_id, data in block_data.items():
        if root in data["nodes"]:
            root_block = block_id
            break

    return bct, root_block, block_data


def _get_cycle_order(G, comp_nodes, start_node):
    """Walk a cycle subgraph from start_node, returning an ordered node list."""
    sg = G.subgraph(comp_nodes)
    order = [start_node]
    visited = {start_node}
    current = start_node
    for _ in range(len(comp_nodes) - 1):
        for neighbor in sg.neighbors(current):
            if neighbor not in visited:
                order.append(neighbor)
                visited.add(neighbor)
                current = neighbor
                break
    return order


def _child_departure_angles(arrival_angle, num_children):
    """Compute evenly-spread departure angles opposite the arrival direction."""
    base = arrival_angle + np.pi
    if num_children == 1:
        return [base]
    spread = min(np.pi * 0.8, np.pi * (num_children - 1) / num_children)
    return [
        base + (-spread / 2 + i * spread / (num_children - 1))
        for i in range(num_children)
    ]


def _layout_block_recursive(
    bct, current_block, parent_cut, entry_pos, entry_angle,
    G, rest_length, positions, placed, block_data,
):
    """Recursively place a block's nodes and recurse into child blocks."""
    data = block_data[current_block]
    nodes = data["nodes"]
    block_type = data["type"]

    # --- place nodes in this block ---
    if block_type == "cycle":
        n = len(nodes)
        r = n * rest_length / (2 * np.pi)

        if parent_cut is not None:
            # Center is inward from entry point along entry_angle
            center = entry_pos + r * np.array(
                [np.cos(entry_angle), np.sin(entry_angle)]
            )
            entry_circle_angle = entry_angle + np.pi
            start_node = parent_cut
        else:
            center = np.array([0.0, 0.0])
            start_node = min(nodes)
            entry_circle_angle = 0.0

        order = _get_cycle_order(G, nodes, start_node)
        angle_step = 2 * np.pi / n

        for i, node in enumerate(order):
            if node not in placed:
                angle = entry_circle_angle + i * angle_step
                positions[node] = center + r * np.array(
                    [np.cos(angle), np.sin(angle)]
                )
                placed.add(node)

        # Recurse into children via cut vertices
        for bct_neighbor in bct.neighbors(current_block):
            if bct_neighbor[0] != "cut":
                continue
            cut_node = bct_neighbor[1]
            if cut_node == parent_cut:
                continue

            # Departure direction: radially outward from cycle center
            outward = positions[cut_node] - center
            arrival_at_cut = np.arctan2(outward[1], outward[0])

            child_blocks = [
                b for b in bct.neighbors(bct_neighbor)
                if b != current_block
            ]
            dep_angles = _child_departure_angles(
                arrival_at_cut + np.pi, len(child_blocks)
            )
            for child_block, dep_angle in zip(child_blocks, dep_angles):
                _layout_block_recursive(
                    bct, child_block, cut_node,
                    positions[cut_node], dep_angle,
                    G, rest_length, positions, placed, block_data,
                )

    else:  # bridge (2 nodes, 1 edge)
        if parent_cut is not None:
            other = (nodes - {parent_cut}).pop()
            if other not in placed:
                direction = np.array(
                    [np.cos(entry_angle), np.sin(entry_angle)]
                )
                positions[other] = entry_pos + rest_length * direction
                placed.add(other)
            exit_node = other
        else:
            # Root is a bridge — place both nodes along x-axis
            sorted_nodes = sorted(nodes)
            positions[sorted_nodes[0]] = np.array([0.0, 0.0])
            placed.add(sorted_nodes[0])
            positions[sorted_nodes[1]] = np.array([rest_length, 0.0])
            placed.add(sorted_nodes[1])
            exit_node = sorted_nodes[1]

        # Recurse into children via cut vertices
        for bct_neighbor in bct.neighbors(current_block):
            if bct_neighbor[0] != "cut":
                continue
            cut_node = bct_neighbor[1]
            if cut_node == parent_cut:
                continue

            child_blocks = [
                b for b in bct.neighbors(bct_neighbor)
                if b != current_block
            ]
            dep_angles = _child_departure_angles(
                entry_angle + np.pi, len(child_blocks)
            )
            for child_block, dep_angle in zip(child_blocks, dep_angles):
                _layout_block_recursive(
                    bct, child_block, cut_node,
                    positions[cut_node], dep_angle,
                    G, rest_length, positions, placed, block_data,
                )


def layout_positions_cactus(undirected_G, root, rest_length):
    """Structure-aware layout for cactus graphs.

    Decomposes into biconnected components, builds a block-cut tree, then
    recursively places cycles as regular polygons and bridges as straight
    segments.  Returns an (N, 2) numpy array centered at the origin.
    """
    num_nodes = undirected_G.number_of_nodes()
    positions = np.zeros((num_nodes, 2))
    placed = set()

    bct, root_block, block_data = _build_block_cut_tree(undirected_G, root)

    # Place root node at origin before recursing
    positions[root] = np.array([0.0, 0.0])
    placed.add(root)

    _layout_block_recursive(
        bct, root_block, None,
        positions[root], 0.0,
        undirected_G, rest_length, positions, placed, block_data,
    )

    # Center at origin
    positions -= positions.mean(axis=0)
    return positions
