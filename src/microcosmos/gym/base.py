import functools
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from microcosmos.simulate import step as _step
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.structs.fields import Fields
from microcosmos.utils import displacement
from microcosmos.solver.config import ConstraintSolverConfig, PBD_SCHEME


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=[
        "nodes",
        "edges",
        "fields",
        "time",
        "energy_center",
        "base_rest_lengths",
        "base_bending_rest_angles",
    ],
)
@dataclass
class EnvState:
    nodes: Nodes
    edges: Edges
    fields: Fields
    time: jax.Array  # () int32
    energy_center: jax.Array  # (2,) float32 -- position of the Gaussian energy source
    base_rest_lengths: jax.Array
    base_bending_rest_angles: jax.Array


class Environment:
    """Pure-functional gymnax-style base environment.

    Subclasses override `_init_state` and the `num_edges` / `num_bending_pairs`
    properties. `reset` and `step` are pure: state flows in and out, nothing is
    stored on self between calls, so `jax.jit` and `jax.vmap` apply directly.

    There is a Gaussian energy source on the grid whose centre does a Brownian
    walk each step. The per-step reward is the total Gaussian value sampled at
    each node's position — the body must chase the source to feed.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (128, 128),
        dt: float = 0.01,
        max_steps: int = 1000,
        solver_config: ConstraintSolverConfig = PBD_SCHEME,
        energy_size: float = 15.0,
        energy_amplitude: float = 0.0,
        energy_walk_std: float = 0.0,
    ):
        """Create a pure-functional environment.

        The default has no food/energy target. Set ``energy_amplitude`` above
        zero for energy-seeking tasks.
        """
        self.grid_shape = grid_shape
        self.dt = dt
        self.max_steps = max_steps
        self.solver_config = solver_config
        self.energy_size = energy_size
        self.energy_amplitude = energy_amplitude
        self.energy_walk_std = energy_walk_std

    # ----- subclass hooks -----

    def _init_state(self, key: jax.Array) -> EnvState:
        raise NotImplementedError

    @property
    def num_edges(self) -> int:
        raise NotImplementedError

    @property
    def num_bending_pairs(self) -> int:
        raise NotImplementedError

    # ----- energy helpers -----

    def _energy_at(self, positions: jax.Array, center: jax.Array) -> jax.Array:
        # Periodic-aware Gaussian: use min-image distance so chasing across
        # the grid boundary still gives a smooth gradient.
        delta = displacement(self.grid_shape, positions, center)
        dist_sq = jnp.sum(delta * delta, axis=-1)
        return self.energy_amplitude * jnp.exp(-dist_sq / (2.0 * self.energy_size ** 2))

    def _sample_energy_center(self, key: jax.Array) -> jax.Array:
        # Fixed at the top-left corner of the world. Screen convention:
        # y increases downward, so "top" = small y, "left" = small x.
        del key
        margin = 2.0 * self.energy_size
        return jnp.array([margin, margin], dtype=jnp.float32)

    # ----- overridable defaults -----

    def _observation(self, state: EnvState) -> jax.Array:
        # COM-centred positions, velocities, and per-node energy reading.
        positions = state.nodes.position
        centered = positions - positions.mean(axis=0)
        energy_at_nodes = self._energy_at(positions, state.energy_center)
        return jnp.concatenate([
            centered.reshape(-1),
            state.nodes.velocity.reshape(-1),
            energy_at_nodes,
        ])

    def _reward(self, prev_state: EnvState, state: EnvState, action) -> jax.Array:
        return jnp.sum(self._energy_at(state.nodes.position, state.energy_center))

    # ----- public API -----

    def action_spec(self) -> dict:
        return {
            "d_rest_length": (self.num_edges,),
            "d_bending_angle": (self.num_bending_pairs,),
        }

    def zero_action(self) -> dict:
        return {
            "d_rest_length": jnp.zeros(self.num_edges),
            "d_bending_angle": jnp.zeros(self.num_bending_pairs),
        }

    def reset(self, key: jax.Array) -> tuple[jax.Array, EnvState]:
        k_state, k_energy = jax.random.split(key)
        state = self._init_state(k_state)
        state = replace(
            state,
            energy_center=self._sample_energy_center(k_energy),
            base_rest_lengths=state.edges.rest_lengths,
            base_bending_rest_angles=state.edges.bending_rest_angles,
        )
        return self._observation(state), state

    def render(self, state: EnvState, size: int = 512) -> np.ndarray:
        from microcosmos.rendering import render_fields
        # Splat the Gaussian onto a grid so the renderer can show it.
        h, w = self.grid_shape
        Y, X = jnp.meshgrid(jnp.arange(h, dtype=jnp.float32), jnp.arange(w, dtype=jnp.float32), indexing="ij")
        grid_pos = jnp.stack([X, Y], axis=-1)
        energy_grid = self._energy_at(grid_pos, state.energy_center)

        nodes_ts = jax.tree.map(lambda x: x[None], state.nodes)
        fields_ts = Fields(
            grid_shape=state.fields.grid_shape,
            steric=state.fields.steric[None],
            fluid_velocity=state.fields.fluid_velocity[None],
            f_grid=None,
            energy=energy_grid[None],
        )
        return render_fields(fields_ts, nodes_ts, sz=size, animate_energy=True)[0]

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: dict,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]:
        new_edges = replace(
            state.edges,
            rest_lengths=state.base_rest_lengths + action["d_rest_length"],
            bending_rest_angles=state.base_bending_rest_angles + action["d_bending_angle"],
        )
        new_nodes, new_edges, new_fields = _step(
            state.nodes, new_edges, state.fields, self.dt, self.solver_config
        )

        # Random walk the energy centre, wrapping around the periodic grid.
        h, w = self.grid_shape
        step_noise = jax.random.normal(key, (2,)) * self.energy_walk_std
        new_center = (state.energy_center + step_noise) % jnp.array([w, h], dtype=state.energy_center.dtype)

        new_state = EnvState(
            nodes=new_nodes,
            edges=new_edges,
            fields=new_fields,
            time=state.time + 1,
            energy_center=new_center,
            base_rest_lengths=state.base_rest_lengths,
            base_bending_rest_angles=state.base_bending_rest_angles,
        )
        obs = self._observation(new_state)
        reward = self._reward(state, new_state, action)
        done = new_state.time >= self.max_steps
        info = {"steps": new_state.time, "energy_center": new_center}
        return obs, new_state, reward, done, info
