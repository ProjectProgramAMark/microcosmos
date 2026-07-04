import sys
from dataclasses import replace

import jax.numpy as jnp

from microcosmos.simulate import step
from microcosmos.rendering import animate_realtime
from microcosmos.solver.config import PBD_SCHEME
from microcosmos.graph import make_edges, compute_bending_pairs
from microcosmos.utils import displacement
from microcosmos.env import MicrocosmosEnv
from experiments.swim_sweep.swimmers.swim_base import Swimmer


class Squirt(Swimmer):
    """Jet propulsion via a pure contracting arc — no mouth sections.

    The entire chain is the reservoir, bent into a ~330° arc (11π/6).
    This is 1/6 of a circle more than the 270° half-loop, slightly closer
    to a full loop.
    """

    plot_color = 'steelblue'
    swimmer_name = 'squirt'

    def setup(self):
        self.num_nodes = self.cfg.experiment.num_nodes
        self.grid_shape = tuple(self.cfg.experiment.grid_shape)
        self.init_line_distance = self.cfg.experiment.get('init_line_distance', 2.0)
        self.frequency = self.cfg.experiment.get('frequency', 0.05)
        self.rest_length_mod_mag = self.cfg.experiment.get('rest_length_mod_mag', 0.0)
        self.enable_animations = self.cfg.experiment.get('enable_animations', False)

        N = self.num_nodes
        d = self.init_line_distance

        edge_pairs, edge_departure_angles = make_edges(
            N, graph="line", num_instances=1
        )
        num_edges = edge_pairs.shape[0]

        domain_height, domain_width = self.grid_shape
        size = tuple(self.grid_shape)

        # Entire chain is the reservoir — 330° arc (11π/6), 1/6 of a circle
        # more than the 270° (3π/2) reservoir-only arc.
        total_arc = 11 * jnp.pi / 6  # 330°
        n_bp = N - 2
        res_turn = -total_arc / max(n_bp, 1)

        bending_rest_angles_shape = jnp.full(n_bp, res_turn)

        # Start direction so arc is symmetric about the vertical axis.
        initial_dir = total_arc / 2  # 11π/12 = 165°
        edge_theta_init = jnp.concatenate([
            jnp.array([initial_dir]),
            initial_dir + jnp.cumsum(bending_rest_angles_shape),
        ])

        dxdy = d * jnp.stack([jnp.cos(edge_theta_init), jnp.sin(edge_theta_init)], axis=1)
        raw_positions = jnp.concatenate([jnp.zeros((1, 2)), jnp.cumsum(dxdy, axis=0)], axis=0)

        cx, cy = domain_width / 2, domain_height / 2
        positions = raw_positions - jnp.mean(raw_positions, axis=0) + jnp.array([cx, cy])

        src, tgt = edge_pairs[:, 0], edge_pairs[:, 1]
        deltas = displacement(size, positions[tgt], positions[src])
        edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

        bending_pairs, _ = compute_bending_pairs(edge_pairs, edge_departure_angles, N)

        raw_diff = edge_theta[1:] - edge_theta[:-1]
        bending_rest_angles = (raw_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi

        # Middle quarter stays fixed; the rest (outer three-quarters) contracts.
        mid = num_edges // 2
        q = num_edges // 4
        mid_start = mid - q // 2
        mid_end = mid_start + q
        self.reservoir_edge_indices = jnp.concatenate([
            jnp.arange(0, mid_start),
            jnp.arange(mid_end, num_edges),
        ])
        # Bending pairs in the middle segment: pair b connects edges b and b+1,
        # so pairs fully inside the middle have b in [mid_start, mid_end-2].
        n_bending = bending_rest_angles.shape[0]
        self.middle_bending_indices = jnp.arange(
            max(mid_start, 0), min(mid_end - 1, n_bending)
        )
        self.base_bending_rest_angles = bending_rest_angles
        self.curvature_boost = self.cfg.experiment.get('curvature_boost', 0.1)

        self.num_edges = num_edges

        self.nodes = self._make_nodes(positions, size)
        self.edges = self._make_edges(
            edge_pairs, edge_theta, num_edges,
            bending_pairs, bending_rest_angles,
        )
        self.fields = self._make_fields()

    def compute_rest_lengths_at_time(self, t: float) -> jnp.ndarray:
        """Fast contraction → hold → slow relaxation. Only reservoir edges contract."""
        period = 1.0 / self.frequency
        phase = (t % period) / period

        fast_fraction = 0.2
        t2 = fast_fraction + 0.1

        if phase < fast_fraction:
            scale = 1.0 - self.rest_length_mod_mag * (phase / fast_fraction)
        elif phase < t2:
            scale = 1.0 - self.rest_length_mod_mag
        else:
            scale = (1.0 - self.rest_length_mod_mag) + self.rest_length_mod_mag * ((phase - t2) / (1.0 - t2))

        rest_lengths = jnp.full(self.num_edges, self.init_line_distance)
        return rest_lengths.at[self.reservoir_edge_indices].set(self.init_line_distance * scale)

    def compute_bending_rest_angles_at_time(self, t: float) -> jnp.ndarray:
        """Increase middle-segment curvature during contraction."""
        period = 1.0 / self.frequency
        phase = (t % period) / period

        fast_fraction = 0.2
        t2 = fast_fraction + 0.1

        if phase < fast_fraction:
            boost = self.curvature_boost * (phase / fast_fraction)
        elif phase < t2:
            boost = self.curvature_boost
        else:
            boost = self.curvature_boost * (1.0 - (phase - t2) / (1.0 - t2))

        # The base angles are negative (curving inward), so subtract to increase curvature
        angles = self.base_bending_rest_angles
        return angles.at[self.middle_bending_indices].add(-boost)

    def reset_simulation(self):
        return self.nodes, self.edges, self._make_fields()

    def run_phase(self, nodes, edges, fields, env: MicrocosmosEnv, phase_name):
        nodes_trajectory = [] if self.enable_animations else None
        fields_trajectory = [] if self.enable_animations else None
        displacement_history = []
        self._init_displacement(nodes)
        print(f"\n{phase_name}:")

        for i in range(env.num_steps):
            t = i * env.dt
            edges = edges.__replace__(
                rest_lengths=self.compute_rest_lengths_at_time(t),
                bending_rest_angles=self.compute_bending_rest_angles_at_time(t),
            )
            nodes, edges, fields = env.step(nodes, edges, fields)

            if jnp.any(jnp.isnan(nodes.position)):
                print(f"Warning: NaN detected at step {i}")
                break

            disp = self._update_displacement(nodes)
            displacement_history.append(disp)
            self._check_stability(nodes, displacement_history)

            if self.enable_animations:
                nodes_trajectory.append(nodes)
                fields_trajectory.append(self._fields_for_render(fields))

            if i % 100 == 0:
                print(f"  Step {i}/{env.num_steps}, Disp: {disp:.4f}", flush=True)
                sys.stdout.flush()

        return nodes, edges, fields, {
            'trajectory': nodes_trajectory,
            'fields_trajectory': fields_trajectory,
            'displacement': displacement_history,
        }

    def run_realtime(self):
        dt = self.cfg.simulation.dt
        viscosity = self.cfg.experiment.realtime_viscousity
        solver_config = replace(PBD_SCHEME, viscosity=viscosity)
        nodes, edges, fields = self.reset_simulation()
        current_time = 0.0
        self._init_displacement(nodes)

        def step_func(nodes, fields, dt, solver_config):
            nonlocal current_time, edges
            edges = edges.__replace__(
                rest_lengths=self.compute_rest_lengths_at_time(current_time),
                bending_rest_angles=self.compute_bending_rest_angles_at_time(current_time),
            )
            nodes, edges_out, fields = step(nodes, edges, fields, dt, solver_config)
            edges = edges_out
            current_time += dt
            return nodes, fields

        animate_realtime(nodes, fields, dt, solver_config, step_func,
                         lambda n, _: [f"Time: {current_time:.2f}",
                                       f"Displacement: {self._update_displacement(n):.4f}"],
                         subtitle="Squirt")
