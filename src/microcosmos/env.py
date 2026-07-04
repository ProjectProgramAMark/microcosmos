import jax.numpy as jnp

from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.structs.fields import Fields
from microcosmos.simulate import simulate, step as _step
from microcosmos.graph import make_edges, compute_bending_pairs, make_fields
from microcosmos.utils import displacement
from microcosmos.solver.config import PBD_SCHEME, PBD_SCHEME_NO_FLUID, ConstraintSolverConfig

DEFAULT_GRID_SHAPE = (64, 64)
DEFAULT_DT = 0.05
DEFAULT_NUM_STEPS = 200
DEFAULT_SPACING = 2.0
DEFAULT_BENDING_STIFFNESS = 0.5


class MicrocosmosEnv:
    """Thin wrapper around the simulator with sensible defaults.

    Experiments instantiate this, call make_chain/make_loop to build initial
    structures, then drive the simulation with step() or rollout().
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
        dt: float = DEFAULT_DT,
        num_steps: int = DEFAULT_NUM_STEPS,
        solver_config: ConstraintSolverConfig = PBD_SCHEME,
    ):
        self.grid_shape = grid_shape
        self.dt = dt
        self.num_steps = num_steps
        self.solver_config = solver_config
        self.fields = make_fields(grid_shape)

    # --- Simulation ---

    def step(self, nodes: Nodes, edges: Edges, fields: Fields | None = None) -> tuple[Nodes, Edges, Fields]:
        """Single physics step. Uses stored fields if none provided."""
        return _step(nodes, edges, fields or self.fields, self.dt, self.solver_config)

    def rollout(self, nodes: Nodes, edges: Edges, fields: Fields | None = None) -> tuple[Nodes, Fields]:
        """Full episode: returns (nodes_timeseries, fields_timeseries)."""
        return simulate(nodes, edges, fields or self.fields, self.dt, self.num_steps, self.solver_config)


    # --- Structure factories ---

    def make_chain(self, num_nodes: int, spacing: float = DEFAULT_SPACING) -> tuple[Nodes, Edges]:
        """Horizontal chain of nodes centered in the grid."""
        h, w = self.grid_shape
        edge_pairs, departure_angles = make_edges(num_nodes, "line")
        bending_pairs, bending_rest_angles = compute_bending_pairs(edge_pairs, departure_angles, num_nodes)

        total_length = (num_nodes - 1) * spacing
        xs = jnp.linspace(w / 2 - total_length / 2, w / 2 + total_length / 2, num_nodes)
        positions = jnp.stack([xs, jnp.full(num_nodes, h / 2.0)], axis=1)

        deltas = displacement(self.grid_shape, positions[edge_pairs[:, 1]], positions[edge_pairs[:, 0]])
        edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

        return self._build(positions, edge_pairs, edge_theta, spacing, bending_pairs, bending_rest_angles)

    def make_loop(self, num_nodes: int, spacing: float = DEFAULT_SPACING) -> tuple[Nodes, Edges]:
        """Closed circular loop of nodes centered in the grid."""
        h, w = self.grid_shape
        edge_pairs, departure_angles = make_edges(num_nodes, "loop")
        bending_pairs, bending_rest_angles = compute_bending_pairs(edge_pairs, departure_angles, num_nodes)

        radius = num_nodes * spacing / (2 * jnp.pi)
        angles = jnp.linspace(0, 2 * jnp.pi, num_nodes, endpoint=False)
        positions = jnp.stack([
            w / 2 + radius * jnp.cos(angles),
            h / 2 + radius * jnp.sin(angles),
        ], axis=1)

        deltas = displacement(self.grid_shape, positions[edge_pairs[:, 1]], positions[edge_pairs[:, 0]])
        edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

        return self._build(positions, edge_pairs, edge_theta, spacing, bending_pairs, bending_rest_angles)

    def _build(self, positions, edge_pairs, edge_theta, spacing, bending_pairs, bending_rest_angles) -> tuple[Nodes, Edges]:
        num_nodes = positions.shape[0]
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
            bending_stiffness=jnp.full(bending_pairs.shape[0], DEFAULT_BENDING_STIFFNESS),
        )
        return nodes, edges

    # --- Presets ---

    @classmethod
    def no_fluid(
        cls,
        grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
        dt: float = DEFAULT_DT,
        num_steps: int = DEFAULT_NUM_STEPS,
    ) -> "MicrocosmosEnv":
        return cls(grid_shape=grid_shape, dt=dt, num_steps=num_steps, solver_config=PBD_SCHEME_NO_FLUID)
