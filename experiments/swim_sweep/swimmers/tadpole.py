import sys

import jax.numpy as jnp

from microcosmos.simulate import step
from microcosmos.rendering import animate_realtime
from microcosmos.solver.config import PBD_SCHEME
from microcosmos.graph import make_edges, compute_bending_pairs
from microcosmos.graph_builder import GraphBuilder
from microcosmos.utils import displacement
from microcosmos.env import MicrocosmosEnv
from experiments.swim_sweep.swimmers.swim_base import Swimmer


class Tadpole(Swimmer):
    """A tadpole with a rolling sine wave in its tail for swimming locomotion."""

    swimmer_name = 'tadpole'

    def setup(self):
        self.num_nodes = self.cfg.experiment.num_nodes
        self.num_wavelengths = self.cfg.experiment.get('num_wavelengths', 1)
        self.rotation_speed = self.cfg.experiment.get('rotation_speed', 0.05)
        self.grid_shape = tuple(self.cfg.experiment.grid_shape)
        self.init_line_distance = self.cfg.experiment.get('init_line_distance', 2.0)
        self.enable_animations = self.cfg.experiment.get('enable_animations', False)

        loop_size = int(self.num_nodes * 0.4)
        tail_size = self.num_nodes - loop_size

        builder = GraphBuilder()
        loop = builder.add_loop(loop_size)
        branch_point = loop[0]
        tail = builder.add_branch(from_node=branch_point, length=tail_size)
        edge_pairs, edge_departure_angles = builder.build()
        num_edges = edge_pairs.shape[0]

        self.tail_nodes = jnp.array(tail)
        self.loop_nodes = jnp.array(loop)

        domain_height, domain_width = self.grid_shape
        loop_radius = (loop_size * self.init_line_distance) / (2 * jnp.pi)
        loop_center_x = domain_width * 0.75
        loop_center_y = domain_height / 2

        loop_angles = jnp.linspace(jnp.pi, 3 * jnp.pi, loop_size, endpoint=False)
        loop_x = loop_center_x + loop_radius * jnp.cos(loop_angles)
        loop_y = loop_center_y + loop_radius * jnp.sin(loop_angles)

        tail_x = jnp.linspace(loop_center_x - loop_radius - self.init_line_distance, domain_width * 0.2, tail_size)
        tail_y = jnp.full(tail_size, loop_center_y)

        positions = jnp.zeros((self.num_nodes, 2))
        positions = positions.at[self.loop_nodes].set(jnp.stack([loop_x, loop_y], axis=1))
        positions = positions.at[self.tail_nodes].set(jnp.stack([tail_x, tail_y], axis=1))

        size = tuple(self.grid_shape)
        src, tgt = edge_pairs[:, 0], edge_pairs[:, 1]
        deltas = displacement(size, positions[tgt], positions[src])
        initial_edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

        bending_pairs, _ = compute_bending_pairs(edge_pairs, edge_departure_angles, self.num_nodes)
        num_bending = bending_pairs.shape[0]

        loop_set, tail_set = set(loop), set(tail)
        loop_bending_indices, tail_bending_indices = [], []
        for bp_idx in range(num_bending):
            e_in, e_out = int(bending_pairs[bp_idx, 0]), int(bending_pairs[bp_idx, 1])
            shared_node = int(edge_pairs[e_in, 1])
            src_in = int(edge_pairs[e_in, 0])
            tgt_out = int(edge_pairs[e_out, 1])
            if src_in in loop_set and shared_node in loop_set and tgt_out in loop_set:
                loop_bending_indices.append(bp_idx)
            elif shared_node in tail_set or shared_node == branch_point:
                tail_bending_indices.append(bp_idx)

        self.loop_bending_indices = jnp.array(loop_bending_indices)
        self.tail_bending_indices = jnp.array(tail_bending_indices)

        amplitude = domain_height * 0.2
        k = 2 * jnp.pi * self.num_wavelengths / jnp.abs(tail_x[-1] - tail_x[0])
        self.initial_phase = 0.0
        self.amplitude = amplitude
        self.k = k
        self.tail_x_positions = tail_x

        bending_rest_angles = jnp.zeros(num_bending)
        bending_rest_angles = bending_rest_angles.at[self.loop_bending_indices].set(2 * jnp.pi / loop_size)
        bending_rest_angles = bending_rest_angles.at[self.tail_bending_indices].set(
            self._compute_tail_bending_curvatures(jnp.arctan(amplitude * k * jnp.cos(k * tail_x + self.initial_phase)))
        )

        self.nodes = self._make_nodes(positions, size)
        self.edges = self._make_edges(edge_pairs, initial_edge_theta, num_edges,
                                      bending_pairs, bending_rest_angles)
        self.fields = self._make_fields()

    def _compute_tail_bending_curvatures(self, target_thetas: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([jnp.array([-jnp.pi / 2]), target_thetas[1:] - target_thetas[:-1]])

    def compute_tail_curvatures_at_phase(self, phase: float) -> jnp.ndarray:
        y_prime = self.amplitude * self.k * jnp.cos(self.k * self.tail_x_positions + phase)
        return self._compute_tail_bending_curvatures(jnp.arctan(y_prime))

    def make_control_fn(self, env: MicrocosmosEnv):
        rotation_speed = self.rotation_speed
        initial_phase = self.initial_phase
        tail_bending_indices = self.tail_bending_indices
        compute = self.compute_tail_curvatures_at_phase

        def control_fn(nodes, edges, t):
            phase = t * rotation_speed + initial_phase
            new_bending = edges.bending_rest_angles.at[tail_bending_indices].set(compute(phase))
            return edges.__replace__(bending_rest_angles=new_bending)

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
            new_bending = edges.bending_rest_angles.at[self.tail_bending_indices].set(
                self.compute_tail_curvatures_at_phase(current_phase)
            )
            edges = edges.__replace__(bending_rest_angles=new_bending)
            nodes, edges_out, fields = step(nodes, edges, fields, dt, solver_config)
            edges = edges_out
            current_phase += self.rotation_speed
            return nodes, fields

        def extra_info_func(nodes, fields):
            disp = self._update_displacement(nodes)
            return [f"Phase: {current_phase:.2f} rad", f"Displacement: {disp:.4f}"]

        animate_realtime(self.nodes, self.fields, dt, PBD_SCHEME, step_func, extra_info_func,
                         subtitle=self.cfg.experiment.type)
