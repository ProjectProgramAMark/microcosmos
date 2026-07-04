import sys

import jax.numpy as jnp
import math

from microcosmos.simulate import step
from microcosmos.rendering import animate_realtime
from microcosmos.solver.config import PBD_SCHEME
from microcosmos.graph import make_edges, compute_bending_pairs
from microcosmos.utils import displacement
from microcosmos.env import MicrocosmosEnv
from experiments.swim_sweep.swimmers.swim_base import Swimmer


class Worm(Swimmer):
    """A rolling sine wave - phase rotates each step by updating local curvatures."""

    swimmer_name = 'worm'

    def setup(self):
        self.num_nodes = math.floor(self.cfg.experiment.num_nodes * 0.6)
        self.num_wavelengths = self.cfg.experiment.get('num_wavelengths', 3)
        self.rotation_speed = self.cfg.experiment.get('rotation_speed', 0.05)
        self.grid_shape = tuple(self.cfg.experiment.grid_shape)
        self.init_line_distance = self.cfg.experiment.get('init_line_distance', 2.0)
        self.enable_animations = self.cfg.experiment.get('enable_animations', False)

        edge_pairs, edge_departure_angles = make_edges(
            self.num_nodes, graph="line", num_instances=1
        )
        num_edges = self.num_nodes - 1

        domain_height, domain_width = self.grid_shape
        x_positions = jnp.linspace(domain_width * 0.1, domain_width * 0.9, self.num_nodes)
        amplitude = domain_height * 0.4
        k = 2 * jnp.pi * self.num_wavelengths / (x_positions[-1] - x_positions[0])

        y_positions = domain_height / 2 + amplitude * jnp.sin(k * x_positions)
        target_thetas = jnp.arctan(amplitude * k * jnp.cos(k * x_positions))
        positions = jnp.stack([x_positions, y_positions], axis=1)

        size = tuple(self.grid_shape)
        src, tgt = edge_pairs[:, 0], edge_pairs[:, 1]
        deltas = displacement(size, positions[tgt], positions[src])
        initial_edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

        bending_pairs, _ = compute_bending_pairs(edge_pairs, edge_departure_angles, self.num_nodes)
        init_bending_rest_angles = target_thetas[1:-1] - target_thetas[:-2]

        self.initial_phase = 0.0
        self.amplitude = amplitude
        self.k = k
        self.x_positions = x_positions

        self.nodes = self._make_nodes(positions, size)
        self.edges = self._make_edges(edge_pairs, initial_edge_theta, num_edges,
                                      bending_pairs, init_bending_rest_angles)
        self.fields = self._make_fields()

    def compute_curvatures_at_phase(self, phase: float) -> jnp.ndarray:
        y_prime = self.amplitude * self.k * jnp.cos(self.k * self.x_positions + phase)
        target_thetas = jnp.arctan(y_prime)
        return target_thetas[1:-1] - target_thetas[:-2]

    def make_control_fn(self, env: MicrocosmosEnv):
        rotation_speed = self.rotation_speed
        initial_phase = self.initial_phase
        compute = self.compute_curvatures_at_phase

        def control_fn(nodes, edges, t):
            phase = t * rotation_speed + initial_phase
            return edges.__replace__(bending_rest_angles=compute(phase))

        return control_fn

    def reset_simulation(self):
        return self.nodes, self.edges, self._make_fields()

    def run_phase(self, nodes, edges, fields, env: MicrocosmosEnv, phase_name):
        control_fn = self.make_control_fn(env)
        nodes_trajectory = [] if self.enable_animations else None
        fields_trajectory = [] if self.enable_animations else None
        displacement_history = []
        self._init_displacement(nodes)

        print(f"\n{phase_name}:")
        for i in range(env.num_steps):
            edges = control_fn(nodes, edges, i)
            nodes, edges, fields = env.step(nodes, edges, fields)
            disp = self._update_displacement(nodes)
            displacement_history.append(disp)
            self._check_stability(nodes, displacement_history)

            if self.enable_animations:
                nodes_trajectory.append(nodes)
                fields_trajectory.append(self._fields_for_render(fields))

            if i % 100 == 0:
                phase = i * self.rotation_speed + self.initial_phase
                print(f"  Step {i}/{env.num_steps}, Phase: {phase:.2f} rad, Disp: {disp:.4f}", flush=True)
                sys.stdout.flush()

        return nodes, edges, fields, {
            'trajectory': nodes_trajectory,
            'fields_trajectory': fields_trajectory,
            'displacement': displacement_history,
        }

    def run_realtime(self):
        dt = self.cfg.simulation.dt
        self._init_displacement(self.nodes)
        current_phase = self.initial_phase
        edges = self.edges

        def step_func(nodes, fields, dt, solver_config):
            nonlocal current_phase, edges
            edges = edges.__replace__(bending_rest_angles=self.compute_curvatures_at_phase(current_phase))
            nodes, edges_out, fields = step(nodes, edges, fields, dt, solver_config)
            edges = edges_out
            current_phase += self.rotation_speed
            return nodes, fields

        def extra_info_func(nodes, fields):
            disp = self._update_displacement(nodes)
            return [f"Phase: {current_phase:.2f} rad", f"Displacement: {disp:.4f}"]

        animate_realtime(self.nodes, self.fields, dt, PBD_SCHEME, step_func, extra_info_func,
                         subtitle=self.cfg.experiment.type)
