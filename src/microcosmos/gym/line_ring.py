import jax
import jax.numpy as jnp

from .multi_agent import LineTopology, MultiAgentEnv, RingTopology


class LineRingEnv(MultiAgentEnv):
    """Compatibility wrapper for the default line-plus-ring multi-agent setup."""

    def __init__(
        self,
        line_topology: LineTopology | None = None,
        ring_topology: RingTopology | None = None,
        line_nodes: int = 16,
        ring_nodes: int = 16,
        spacing: float = 2.0,
        bending_stiffness: float = 0.5,
        centered_line: bool = False,
        ring_spawn_radius: float | None = None,
        random_ring_angle: bool = False,
        **kwargs,
    ):
        line = line_topology or LineTopology(
            num_nodes=line_nodes,
            spacing=spacing,
            bending_stiffness=bending_stiffness,
        )
        ring = ring_topology or RingTopology(
            num_nodes=ring_nodes,
            spacing=spacing,
            bending_stiffness=bending_stiffness,
        )
        self._centered_line = centered_line
        self._ring_spawn_radius = ring_spawn_radius
        self._random_ring_angle = random_ring_angle
        super().__init__(creatures=(line, ring), **kwargs)

    def _agent_centers(self, key: jax.Array) -> jax.Array:
        if not self._centered_line:
            return super()._agent_centers(key)

        h, w = self.grid_shape
        spawn_r = self._ring_spawn_radius
        if spawn_r is None:
            spawn_r = w / 4.0
        if self._random_ring_angle:
            spawn_angle = jax.random.uniform(key, (), minval=0.0, maxval=2 * jnp.pi)
        else:
            spawn_angle = jnp.array(0.0)

        line_center = jnp.array([w / 2.0, h / 2.0])
        ring_center = jnp.array(
            [
                w / 2.0 + spawn_r * jnp.cos(spawn_angle),
                h / 2.0 + spawn_r * jnp.sin(spawn_angle),
            ]
        )
        return jnp.stack([line_center, ring_center], axis=0)
