import jax
import jax.numpy as jnp

from microcosmos.graph import make_edges, compute_bending_pairs, make_fields
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.utils import displacement

from .base import Environment, EnvState


_BENDING_STIFFNESS = 0.5


class RingEnv(Environment):
    """Closed circular loop of nodes."""

    def __init__(self, num_nodes: int = 16, spacing: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self._num_nodes = num_nodes
        self._spacing = spacing

        self._edge_pairs, self._departure_angles = make_edges(num_nodes, "loop")
        self._bending_pairs, self._bending_rest_angles = compute_bending_pairs(
            self._edge_pairs, self._departure_angles, num_nodes
        )
        self._num_edges = int(self._edge_pairs.shape[0])
        self._num_bending_pairs = int(self._bending_pairs.shape[0])

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @property
    def num_bending_pairs(self) -> int:
        return self._num_bending_pairs

    def _init_state(self, key: jax.Array) -> EnvState:
        del key
        h, w = self.grid_shape
        n = self._num_nodes
        spacing = self._spacing

        edge_pairs = self._edge_pairs
        bending_pairs = self._bending_pairs
        bending_rest_angles = self._bending_rest_angles

        radius = n * spacing / (2 * jnp.pi)
        angles = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        positions = jnp.stack(
            [w / 2 + radius * jnp.cos(angles), h / 2 + radius * jnp.sin(angles)],
            axis=1,
        )

        deltas = displacement(
            self.grid_shape, positions[edge_pairs[:, 1]], positions[edge_pairs[:, 0]]
        )
        edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

        nodes = Nodes(
            position=positions,
            velocity=jnp.zeros_like(positions),
            color_bend=jnp.arange(n) % 3,
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
