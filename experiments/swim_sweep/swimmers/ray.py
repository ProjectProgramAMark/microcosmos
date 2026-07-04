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


class Ray(Swimmer):
    """Demonstrate the Scallop Theorem using a gliding (Fast/Hold/Slow) strategy."""

    plot_color = 'orangered'
    swimmer_name = 'ray'

    def setup(self):
        self.num_nodes = self.cfg.experiment.num_nodes
        self.grid_shape = tuple(self.cfg.experiment.grid_shape)
        self.init_line_distance = self.cfg.experiment.get('init_line_distance', 2.0)
        self.frequency = self.cfg.experiment.get('frequency', 0.1)
        self.amplitude = self.cfg.experiment.get('amplitude', 0.45)
        self.hinge_position = self.cfg.experiment.get('hinge_position', 0.5)
        self.enable_animations = self.cfg.experiment.get('enable_animations', False)

        edge_pairs, edge_departure_angles = make_edges(
            self.num_nodes, graph="line", num_instances=1
        )
        num_edges = edge_pairs.shape[0]

        domain_height, domain_width = self.grid_shape
        size = tuple(self.grid_shape)

        positions = jnp.stack([
            jnp.full(self.num_nodes, domain_width / 2),
            jnp.linspace(domain_height * 0.2, domain_height * 0.8, self.num_nodes),
        ], axis=1)

        src, tgt = edge_pairs[:, 0], edge_pairs[:, 1]
        deltas = displacement(size, positions[tgt], positions[src])
        edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

        bending_pairs, bending_rest_angles = compute_bending_pairs(
            edge_pairs, edge_departure_angles, self.num_nodes
        )

        self.nodes = self._make_nodes(positions, size)
        self.edges = self._make_edges(edge_pairs, edge_theta, num_edges,
                                      bending_pairs, bending_rest_angles)
        self.fields = self._make_fields()

    def compute_curvature_at_time(self, t: float) -> jnp.ndarray:
        """Compute hinge curvature: close 180°→30° in 1/4 period, hold 1/4, open back to 180° in 1/2."""
        num_bending_pairs = self.num_nodes - 2
        period = 1.0 / self.frequency
        phase = (t % period) / period

        # Timing: close π/2, hold π/2, open π (of 2π total)
        close_fraction = 0.25
        hold_fraction = 0.25
        t1 = close_fraction
        t2 = close_fraction + hold_fraction

        # Signal: 0 (open/180°) → 1 (closed/30°) → 0 (open/180°)
        if phase < t1:
            signal = phase / close_fraction
        elif phase < t2:
            signal = 1.0
        else:
            signal = 1.0 - (phase - t2) / (1.0 - t2)

        # Total hinge bend: 0 at 180° to 150° (= 5π/6 rad) at 30°
        max_total_curvature = (180.0 - 30.0) * jnp.pi / 180.0

        # Distribute curvature around hinge with Gaussian weighting,
        # normalized so the sum of bending angles equals the target total
        hinge_idx = int(self.hinge_position * self.num_nodes)
        bending_angles = jnp.zeros(num_bending_pairs)
        spread = 3

        total_weight = 0.0
        for offset in range(-spread, spread + 1):
            idx = hinge_idx + offset
            if 0 <= idx < num_bending_pairs:
                total_weight += float(jnp.exp(-offset**2 / (2 * (spread / 2) ** 2)))

        for offset in range(-spread, spread + 1):
            idx = hinge_idx + offset
            if 0 <= idx < num_bending_pairs:
                weight = jnp.exp(-offset**2 / (2 * (spread / 2) ** 2))
                bending_angles = bending_angles.at[idx].add(
                    max_total_curvature * signal * weight / total_weight
                )

        return bending_angles

    def reset_simulation(self):
        return self.nodes, self.edges, self._make_fields()

    def run_phase(self, nodes, edges, fields, env: MicrocosmosEnv, phase_name):
        nodes_trajectory = [] if self.enable_animations else None
        fields_trajectory = [] if self.enable_animations else None
        curvature_history = []
        displacement_history = []
        self._init_displacement(nodes)
        hinge_idx = int(self.hinge_position * self.num_nodes)

        print(f"\n{phase_name}:")
        for i in range(env.num_steps):
            new_bending_angles = self.compute_curvature_at_time(i * env.dt)
            edges = edges.__replace__(bending_rest_angles=new_bending_angles)
            curvature_history.append(float(new_bending_angles[hinge_idx]))

            nodes, edges, fields = env.step(nodes, edges, fields)

            disp = self._update_displacement(nodes)
            displacement_history.append(disp)
            self._check_stability(nodes, displacement_history)

            if self.enable_animations:
                nodes_trajectory.append(nodes)
                fields_trajectory.append(self._fields_for_render(fields))

            if i % 100 == 0:
                print(f"  Step {i}/{env.num_steps}, Disp: {disp:.4f}, Curv: {new_bending_angles[hinge_idx]:.3f}", flush=True)
                sys.stdout.flush()

        return nodes, edges, fields, {
            'trajectory': nodes_trajectory,
            'fields_trajectory': fields_trajectory,
            'displacement': displacement_history,
            'curvature': curvature_history,
        }

    def run_phase_realtime(self, in_nodes, in_edges, in_fields, dt, phase_name, solver_config):
        self._init_displacement(in_nodes)
        hinge_idx = int(self.hinge_position * self.num_nodes)
        current_time = 0.0
        curvature = 0.0
        edges = in_edges

        def step_func(nodes, fields, *_):
            nonlocal current_time, curvature, edges
            new_bending_angles = self.compute_curvature_at_time(current_time)
            edges = edges.__replace__(bending_rest_angles=new_bending_angles)
            curvature = float(new_bending_angles[hinge_idx])
            nodes, edges, fields = step(nodes, edges, fields, dt, solver_config)
            current_time += dt
            return nodes, fields

        def extra_info_func(nodes, fields):
            disp = self._update_displacement(nodes)
            return [f"Curvature: {curvature:.2f}", f"Displacement: {disp:.2f}"]

        animate_realtime(in_nodes, in_fields, dt, solver_config, step_func, extra_info_func, subtitle=phase_name)

    def run_realtime(self):
        dt = self.cfg.simulation.dt
        viscosity = self.cfg.experiment.realtime_viscousity
        print(f"Running Ray (real-time), viscosity={viscosity}")
        solver_config = replace(PBD_SCHEME, viscosity=viscosity)
        nodes, edges, fields = self.reset_simulation()
        self.run_phase_realtime(nodes, edges, fields, dt, 'Gliding', solver_config)
