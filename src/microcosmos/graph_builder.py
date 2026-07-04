"""
Graph builder using NetworkX for flexible topology specification.

Examples:
    # Simple loop
    >>> builder = GraphBuilder()
    >>> builder.add_loop(50)
    >>> edges, departure_angles = builder.build()

    # Loop with branches
    >>> builder = GraphBuilder()
    >>> loop_nodes = builder.add_loop(50)
    >>> builder.add_branch(from_node=1, length=40)
    >>> builder.add_branch(from_node=10, length=40)
    >>> edges, departure_angles = builder.build()

    # Tree structure
    >>> builder = GraphBuilder()
    >>> trunk = builder.add_line(40)
    >>> builder.add_branch(from_node=trunk[20], length=20)
    >>> builder.add_branch(from_node=trunk[20], length=20)
    >>> edges, departure_angles = builder.build()
"""

import jax.numpy as jnp
import networkx as nx
import numpy as np


class GraphBuilder:
    """
    Builder for creating directed graph topologies using NetworkX.

    The graph stores forward-only directed edges. Each undirected bond
    is stored as ONE directed edge (forward along chain, parent->child
    for branches). Departure angles are stored as edge attributes.
    """

    def __init__(self):
        """Initialize an empty directed graph."""
        self.G = nx.DiGraph()
        self.next_id = 0

    def add_loop(self, length: int) -> list[int]:
        """
        Add a closed loop (ring) of nodes.

        Args:
            length: Number of nodes in the loop

        Returns:
            List of node IDs in the loop
        """
        nodes = list(range(self.next_id, self.next_id + length))
        for i in range(len(nodes)):
            curr = nodes[i]
            next_node = nodes[(i + 1) % len(nodes)]
            self.G.add_edge(curr, next_node, departure_angle=0.0)
        self.next_id += length
        return nodes

    def add_line(self, length: int) -> list[int]:
        """
        Add an open chain (line) of nodes.

        Args:
            length: Number of nodes in the line

        Returns:
            List of node IDs in the line
        """
        nodes = list(range(self.next_id, self.next_id + length))
        for i in range(len(nodes) - 1):
            curr = nodes[i]
            next_node = nodes[i + 1]
            self.G.add_edge(curr, next_node, departure_angle=0.0)
        self.next_id += length
        return nodes

    def add_branch(self, from_node: int, length: int, departure_angle: float | None = None) -> list[int]:
        """
        Add a branch extending from an existing node.

        Args:
            from_node: Node ID to attach the branch to
            length: Number of nodes in the branch

        Returns:
            List of node IDs in the branch
        """
        branch_nodes = list(range(self.next_id, self.next_id + length))
        self.next_id += length

        # Add internal edges for the branch line
        for i in range(len(branch_nodes) - 1):
            curr = branch_nodes[i]
            next_node = branch_nodes[i + 1]
            self.G.add_edge(curr, next_node, departure_angle=0.0)

        # Determine departure angle for the junction edge
        if departure_angle is not None:
            branch_departure = departure_angle
        else:
            # Count existing forward edges from from_node to pick angle
            existing_forward = list(self.G.successors(from_node))
            num_existing_branches = 0
            for neighbor in existing_forward:
                angle = self.G[from_node][neighbor].get('departure_angle', 0.0)
                if abs(angle) > 0.1:  # It's a branch edge (not a chain edge)
                    num_existing_branches += 1

            # First branch gets +pi/2, second gets -pi/2
            if num_existing_branches == 0:
                branch_departure = np.pi / 2.0
            else:
                branch_departure = -np.pi / 2.0

        # Connect parent to branch start
        self.G.add_edge(from_node, branch_nodes[0], departure_angle=branch_departure)

        # Update internal branch edges to carry the same departure angle,
        # so that consecutive edges within a branch have rest angle 0 (straight)
        for i in range(len(branch_nodes) - 1):
            self.G[branch_nodes[i]][branch_nodes[i + 1]]['departure_angle'] = branch_departure

        return branch_nodes

    def build(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Convert the graph to edge list format.

        Returns:
            edges: (E, 2) int32 array of (source, target) pairs
            departure_angles: (E,) float32 array of departure angles
        """
        num_nodes = self.G.number_of_nodes()

        # Verify nodes are contiguous integers starting from 0
        expected_nodes = set(range(num_nodes))
        actual_nodes = set(self.G.nodes())
        if expected_nodes != actual_nodes:
            raise ValueError(
                f"Graph nodes must be contiguous integers from 0 to {num_nodes-1}. "
                f"Got nodes: {sorted(actual_nodes)}"
            )

        # Extract edges and departure angles
        edge_list = []
        dep_angles = []
        for src, tgt, data in self.G.edges(data=True):
            edge_list.append((src, tgt))
            dep_angles.append(data.get('departure_angle', 0.0))

        edges = jnp.array(edge_list, dtype=jnp.int32) if edge_list else jnp.zeros((0, 2), dtype=jnp.int32)
        departure_angles = jnp.array(dep_angles, dtype=jnp.float32) if dep_angles else jnp.zeros(0, dtype=jnp.float32)

        return edges, departure_angles


def load_graph_from_json(
    json_path: str,
    shape: tuple[int, int] = (128, 128),
    rest_length: float = 2.0,
    layout: str = "cactus",
):
    """
    Load an undirected graph from a JSON file and return a ready-to-simulate tuple.

    JSON format::

        {"nodes": [0, 1, 2, ...], "edges": [[0, 1], [1, 2], ...]}

    Args:
        json_path: Path to the JSON file.
        shape: Grid shape (H, W) for the simulation domain.
        rest_length: Target rest length between connected nodes.
        layout: Layout algorithm — "cactus" (default) or "kamada_kawai".

    Returns:
        (nodes, edges, fields) ready to pass to simulate().
    """
    import json
    from pathlib import Path
    from microcosmos.graph import build_simulation_from_undirected

    data = json.loads(Path(json_path).read_text())

    G = nx.Graph()
    for edge in data.get("edges", []):
        G.add_edge(int(edge[0]), int(edge[1]))
    for node in data.get("nodes", []):
        G.add_node(int(node))

    return build_simulation_from_undirected(G, shape=shape, rest_length=rest_length, layout=layout)
