from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp

from microcosmos.graph import compute_bending_pairs, make_edges, make_fields
from microcosmos.structs.edges import Edges
from microcosmos.structs.nodes import Nodes
from microcosmos.utils import displacement

from .base import Environment, EnvState


@dataclass(frozen=True)
class CreatureTopology:
    """Local creature topology and rest geometry."""

    num_nodes: int
    spacing: float = 2.0
    bending_stiffness: float = 0.5

    def edges(self) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    def local_positions(self) -> jax.Array:
        raise NotImplementedError


@dataclass(frozen=True)
class LineTopology(CreatureTopology):
    """Open chain topology centered around the local origin."""

    def edges(self) -> tuple[jax.Array, jax.Array]:
        return make_edges(self.num_nodes, "line")

    def local_positions(self) -> jax.Array:
        total_length = (self.num_nodes - 1) * self.spacing
        xs = jnp.linspace(-total_length / 2.0, total_length / 2.0, self.num_nodes)
        return jnp.stack([xs, jnp.zeros(self.num_nodes)], axis=1)


@dataclass(frozen=True)
class RingTopology(CreatureTopology):
    """Closed loop topology centered around the local origin."""

    def edges(self) -> tuple[jax.Array, jax.Array]:
        return make_edges(self.num_nodes, "loop")

    def local_positions(self) -> jax.Array:
        radius = self.num_nodes * self.spacing / (2 * jnp.pi)
        angles = jnp.linspace(0, 2 * jnp.pi, self.num_nodes, endpoint=False)
        return jnp.stack([radius * jnp.cos(angles), radius * jnp.sin(angles)], axis=1)


class MultiAgentEnv(Environment):
    """Environment containing multiple independent creature topologies."""

    def __init__(
        self,
        creatures: Sequence[CreatureTopology] | None = None,
        random_initial_positions: bool = False,
        position_margin: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if creatures is None:
            creatures = (LineTopology(), RingTopology())
        if not creatures:
            raise ValueError("MultiAgentEnv requires at least one creature topology")
        self._creatures = tuple(creatures)
        self._random_initial_positions = random_initial_positions
        self._position_margin = position_margin

        edge_parts = []
        rest_length_parts = []
        bending_pair_parts = []
        bending_rest_angle_parts = []
        bending_stiffness_parts = []
        local_position_parts = []
        center_margin_parts = []
        node_slices = []
        edge_slices = []
        bending_slices = []
        node_offset = 0
        edge_offset = 0
        bending_offset = 0

        for creature in self._creatures:
            edge_pairs, departure_angles = creature.edges()
            local_positions = creature.local_positions()
            num_edges = int(edge_pairs.shape[0])

            local_bending_pairs, local_bending_rest_angles = compute_bending_pairs(
                edge_pairs, departure_angles, creature.num_nodes
            )
            num_bending_pairs = int(local_bending_pairs.shape[0])

            edge_parts.append(edge_pairs + node_offset)
            rest_length_parts.append(jnp.full(num_edges, creature.spacing))
            bending_pair_parts.append(local_bending_pairs + edge_offset)
            bending_rest_angle_parts.append(local_bending_rest_angles)
            bending_stiffness_parts.append(
                jnp.full(num_bending_pairs, creature.bending_stiffness)
            )
            local_position_parts.append(local_positions)
            center_margin_parts.append(
                jnp.max(jnp.abs(local_positions), axis=0) + position_margin
            )
            node_slices.append(slice(node_offset, node_offset + creature.num_nodes))
            edge_slices.append(slice(edge_offset, edge_offset + num_edges))
            bending_slices.append(
                slice(bending_offset, bending_offset + num_bending_pairs)
            )

            node_offset += creature.num_nodes
            edge_offset += num_edges
            bending_offset += num_bending_pairs

        self._node_slices = tuple(node_slices)
        self._edge_slices = tuple(edge_slices)
        self._bending_slices = tuple(bending_slices)
        self._local_positions = tuple(local_position_parts)
        self._center_margins = jnp.stack(center_margin_parts, axis=0)
        self._edge_pairs = jnp.concatenate(edge_parts, axis=0)
        self._rest_lengths = jnp.concatenate(rest_length_parts, axis=0)
        self._bending_pairs = jnp.concatenate(bending_pair_parts, axis=0)
        self._bending_rest_angles = jnp.concatenate(bending_rest_angle_parts, axis=0)
        self._bending_stiffness = jnp.concatenate(bending_stiffness_parts, axis=0)
        self._num_nodes = node_offset
        self._num_edges = int(self._edge_pairs.shape[0])
        self._num_bending_pairs = int(self._bending_pairs.shape[0])

    @property
    def creatures(self) -> tuple[CreatureTopology, ...]:
        return self._creatures

    @property
    def node_slices(self) -> tuple[slice, ...]:
        return self._node_slices

    @property
    def edge_slices(self) -> tuple[slice, ...]:
        return self._edge_slices

    @property
    def bending_slices(self) -> tuple[slice, ...]:
        return self._bending_slices

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @property
    def num_bending_pairs(self) -> int:
        return self._num_bending_pairs

    def split_action(self, action: dict[str, jax.Array]) -> tuple[dict[str, jax.Array], ...]:
        """Split a flat action vector into one action dict per creature."""
        return tuple(
            {
                "d_rest_length": action["d_rest_length"][edge_slice],
                "d_bending_angle": action["d_bending_angle"][bending_slice],
            }
            for edge_slice, bending_slice in zip(
                self._edge_slices, self._bending_slices, strict=True
            )
        )

    def merge_action(self, creature_actions: Sequence[dict[str, jax.Array]]) -> dict[str, jax.Array]:
        """Merge per-creature action dicts into the flat action expected by step()."""
        if len(creature_actions) != len(self._creatures):
            raise ValueError(
                f"Expected {len(self._creatures)} creature actions, got {len(creature_actions)}"
            )
        return {
            "d_rest_length": jnp.concatenate(
                [action["d_rest_length"] for action in creature_actions], axis=0
            ),
            "d_bending_angle": jnp.concatenate(
                [action["d_bending_angle"] for action in creature_actions], axis=0
            ),
        }

    def fitness_per_creature(self, state: EnvState) -> jax.Array:
        """Return one scalar fitness contribution per creature at this state."""
        per_creature = []
        for node_slice in self._node_slices:
            per_creature.append(
                jnp.sum(self._energy_at(state.nodes.position[node_slice], state.energy_center))
            )
        return jnp.stack(per_creature)

    def _reward(self, prev_state: EnvState, state: EnvState, action) -> jax.Array:
        del prev_state, action
        return jnp.sum(self.fitness_per_creature(state))

    def _agent_centers(self, key: jax.Array) -> jax.Array:
        h, w = self.grid_shape
        num_creatures = len(self._creatures)
        if self._random_initial_positions:
            world = jnp.array([w, h], dtype=jnp.float32)
            margins = jnp.minimum(self._center_margins, world / 2.0)
            span = world - 2.0 * margins
            return margins + jax.random.uniform(key, (num_creatures, 2)) * span

        if num_creatures == 1:
            xs = jnp.array([w / 2.0])
        elif num_creatures == 2:
            xs = jnp.array([w / 4.0, 3.0 * w / 4.0])
        else:
            xs = jnp.linspace(
                w / (num_creatures + 1),
                w * num_creatures / (num_creatures + 1),
                num_creatures,
            )
        return jnp.stack([xs, jnp.full(num_creatures, h / 2.0)], axis=1)

    def _init_state(self, key: jax.Array) -> EnvState:
        centers = self._agent_centers(key)
        positions = jnp.concatenate(
            [
                local_positions + centers[i]
                for i, local_positions in enumerate(self._local_positions)
            ],
            axis=0,
        )

        deltas = displacement(
            self.grid_shape,
            positions[self._edge_pairs[:, 1]],
            positions[self._edge_pairs[:, 0]],
        )
        edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

        nodes = Nodes(
            position=positions,
            velocity=jnp.zeros_like(positions),
            color_bend=jnp.arange(self.num_nodes) % 3,
            debug_vector=jnp.zeros_like(positions),
        )
        edges = Edges(
            pairs=self._edge_pairs,
            theta=edge_theta,
            rest_lengths=self._rest_lengths,
            bending_pairs=self._bending_pairs,
            bending_rest_angles=self._bending_rest_angles,
            bending_stiffness=self._bending_stiffness,
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
