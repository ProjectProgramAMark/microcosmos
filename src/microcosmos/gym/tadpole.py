import jax
import jax.numpy as jnp

from microcosmos.graph import (
    compute_bending_pairs,
    make_fields,
    set_orthogonal_rest_angles,
)
from microcosmos.graph_builder import GraphBuilder
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.utils import displacement

from .base import Environment, EnvState


_BENDING_STIFFNESS = 0.5


def _build_topology(head_nodes: int, tail_nodes: int):
    builder = GraphBuilder()
    head_ids = builder.add_loop(head_nodes)
    builder.add_branch(from_node=head_ids[0], length=tail_nodes)
    return builder.build()


class TadpoleEnv(Environment):
    """Tadpole-shaped body: a circular head loop with a straight tail attached."""

    def __init__(
        self,
        head_nodes: int = 8,
        tail_nodes: int = 10,
        spacing: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._head_nodes = head_nodes
        self._tail_nodes = tail_nodes
        self._spacing = spacing

        self._edge_pairs, self._departure_angles = _build_topology(head_nodes, tail_nodes)
        self._bending_pairs, self._bending_rest_angles = compute_bending_pairs(
            self._edge_pairs, self._departure_angles, head_nodes + tail_nodes
        )
        self._num_edges = int(self._edge_pairs.shape[0])
        self._num_bending_pairs = int(self._bending_pairs.shape[0])

    @property
    def num_nodes(self) -> int:
        return self._head_nodes + self._tail_nodes

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @property
    def num_bending_pairs(self) -> int:
        return self._num_bending_pairs

    def _init_state(self, key: jax.Array) -> EnvState:
        del key
        h, w = self.grid_shape
        head_n = self._head_nodes
        tail_n = self._tail_nodes
        spacing = self._spacing
        num_nodes = head_n + tail_n

        edge_pairs = self._edge_pairs

        radius = head_n * spacing / (2 * jnp.pi)
        # Shift head left so head+tail fits roughly centered.
        cx = w / 2 - (tail_n * spacing) / 2
        cy = h / 2

        head_angles = jnp.linspace(0.0, 2 * jnp.pi, head_n, endpoint=False)
        head_positions = jnp.stack(
            [cx + radius * jnp.cos(head_angles), cy + radius * jnp.sin(head_angles)],
            axis=1,
        )
        # Tail starts one spacing past head node 0 (which sits at angle=0).
        tail_xs = jnp.linspace(
            cx + radius + spacing,
            cx + radius + tail_n * spacing,
            tail_n,
        )
        tail_positions = jnp.stack([tail_xs, jnp.full(tail_n, cy)], axis=1)
        positions = jnp.concatenate([head_positions, tail_positions], axis=0)

        deltas = displacement(
            self.grid_shape, positions[edge_pairs[:, 1]], positions[edge_pairs[:, 0]]
        )
        edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

        bending_pairs = self._bending_pairs
        bending_rest_angles = self._bending_rest_angles
        # Set the head/tail junction's branch pair to ±π/2 based on actual layout geometry.
        bending_rest_angles = set_orthogonal_rest_angles(
            edge_pairs, edge_theta, bending_pairs, bending_rest_angles, num_nodes
        )

        nodes = Nodes(
            position=positions,
            velocity=jnp.zeros_like(positions),
            color_bend=jnp.arange(num_nodes) % 3,
            debug_vector=jnp.zeros_like(positions),
        )
        edges = Edges(
            pairs=edge_pairs,
            theta=edge_theta,
            rest_lengths=jnp.full(edge_pairs.shape[0], spacing),
            bending_pairs=bending_pairs,
            bending_rest_angles=bending_rest_angles,
            bending_stiffness=jnp.full(bending_pairs.shape[0], _BENDING_STIFFNESS),
        )
        return EnvState(
            nodes=nodes,
            edges=edges,
            fields=make_fields(self.grid_shape),
            time=jnp.array(0, dtype=jnp.int32),
            energy_center=jnp.zeros(2, dtype=jnp.float32),
            base_rest_lengths=edges.rest_lengths,
            base_bending_rest_angles=edges.bending_rest_angles,
        )
