import sys

import jax.numpy as jnp
import numpy as np

from microcosmos.simulate import step
from microcosmos.rendering import animate_realtime
from microcosmos.solver.config import PBD_SCHEME
from microcosmos.graph_builder import GraphBuilder
from microcosmos.graph import compute_bending_pairs
from microcosmos.utils import displacement
from microcosmos.env import MicrocosmosEnv
from experiments.swim_sweep.swimmers.swim_base import Swimmer


class Cilia(Swimmer):
    """A spherical bacterium covered in cilia (short beating appendages).

    Body: a loop of nodes forming a circular cell body.
    Cilia: 30 short branches radiating outward, beating in a metachronal wave
    (a traveling wave of phase-shifted power/recovery strokes around the body).

    Stiffness: cilia themselves are floppy (low bending stiffness), but the
    junction where each cilium attaches to the body has normal stiffness so
    the cilia stay anchored at their base angle.
    """

    plot_color = 'mediumorchid'
    swimmer_name = 'cilia'

    BODY_SIZE = 60       # nodes in the circular body
    CILIUM_LENGTH = 10    # nodes per cilium (not counting the body node)
    NUM_CILIA = 15       # one cilium per body node

    def setup(self):
        self.grid_shape = tuple(self.cfg.experiment.grid_shape)
        self.init_line_distance = self.cfg.experiment.get('init_line_distance', 2.0)
        self.rotation_speed = self.cfg.experiment.get('rotation_speed', 6.6)
        self.amplitude = self.cfg.experiment.get('amplitude', 1.5)
        self.enable_animations = self.cfg.experiment.get('enable_animations', False)
        # Number of metachronal waves around the body
        self.num_waves = self.cfg.experiment.get('num_metachronal_waves', 2)

        self.num_nodes = self.BODY_SIZE + self.NUM_CILIA * self.CILIUM_LENGTH

        # Build topology: loop body + 30 cilia branches
        builder = GraphBuilder()
        body_nodes = builder.add_loop(self.BODY_SIZE)

        cilia = []
        stride = self.BODY_SIZE // self.NUM_CILIA
        for node in body_nodes[::stride]:
            cilia.append(builder.add_branch(from_node=node, length=self.CILIUM_LENGTH, departure_angle=-np.pi / 2))

        edge_pairs, edge_departure_angles = builder.build()
        num_edges = edge_pairs.shape[0]

        # Lay out body nodes in a circle
        domain_height, domain_width = self.grid_shape
        center_x, center_y = domain_width / 2.0, domain_height / 2.0
        body_radius = self.BODY_SIZE * self.init_line_distance / (2 * np.pi)

        angles = jnp.linspace(0, 2 * jnp.pi, self.BODY_SIZE, endpoint=False)
        body_x = center_x + body_radius * jnp.cos(angles)
        body_y = center_y + body_radius * jnp.sin(angles)

        positions = jnp.zeros((self.num_nodes, 2))
        positions = positions.at[jnp.array(body_nodes)].set(
            jnp.stack([body_x, body_y], axis=1)
        )

        # Place cilia nodes radiating outward from each body node
        for i, cilium in enumerate(cilia):
            body_idx = i * stride
            angle = float(angles[body_idx])
            direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])
            base = jnp.array([float(body_x[body_idx]), float(body_y[body_idx])])
            for j, node_id in enumerate(cilium):
                pos = base + direction * self.init_line_distance * (j + 1)
                positions = positions.at[node_id].set(pos)

        # Compute initial edge thetas from actual positions
        src, tgt = edge_pairs[:, 0], edge_pairs[:, 1]
        size = tuple(self.grid_shape)
        deltas = displacement(size, positions[tgt], positions[src])
        initial_edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

        bending_pairs, bending_rest_angles = compute_bending_pairs(
            edge_pairs, edge_departure_angles, self.num_nodes
        )

        # Find all bending pair indices for each cilium in order (junction first, then internals)
        self.per_cilium_bp_indices = self._find_per_cilium_bp_indices(
            edge_pairs, cilia, bending_pairs
        )  # shape (NUM_CILIA, bps_per_cilium)

        self.initial_phase = 0.0
        self.nodes = self._make_nodes(positions, size)
        self.edges = self._make_edges(
            edge_pairs, initial_edge_theta, num_edges, bending_pairs, bending_rest_angles
        )

        # Uniform stiffness across all bending pairs
        self.edges = self.edges.__replace__(bending_stiffness=jnp.ones(bending_pairs.shape[0]))

        self.fields = self._make_fields()

        # Cache baseline angles for all cilium BPs: shape (NUM_CILIA, bps_per_cilium)
        self.per_cilium_baseline_angles = self.edges.bending_rest_angles[self.per_cilium_bp_indices]
        # Distribute phase around the body to create metachronal waves
        self.phase_offsets = jnp.linspace(0, 2 * jnp.pi * self.num_waves, self.NUM_CILIA, endpoint=False)
        # Bilateral stroke signs: top half (+y side) sweeps one way, bottom half the other
        cilia_body_angles = jnp.array([
            i * stride * 2 * np.pi / self.BODY_SIZE for i in range(self.NUM_CILIA)
        ])
        self.stroke_signs = jnp.where(cilia_body_angles < np.pi, 1.0, -1.0)

    def _find_per_cilium_bp_indices(self, edge_pairs, cilia, bending_pairs):
        """Find all bending pair indices for each cilium in order, from junction to tip.

        Returns shape (NUM_CILIA, bps_per_cilium) where bps_per_cilium = CILIUM_LENGTH - 1.
        """
        edge_pairs_np = np.array(edge_pairs)
        bending_pairs_np = np.array(bending_pairs)

        edge_lookup = {
            (int(edge_pairs_np[e, 0]), int(edge_pairs_np[e, 1])): e
            for e in range(edge_pairs_np.shape[0])
        }
        bp_lookup = {
            (int(bending_pairs_np[b, 0]), int(bending_pairs_np[b, 1])): b
            for b in range(bending_pairs_np.shape[0])
        }

        per_cilium = []
        for cilium in cilia:
            # Junction edge is the one whose target is the first cilium node
            junction_e = next(
                e for e in range(edge_pairs_np.shape[0])
                if int(edge_pairs_np[e, 1]) == cilium[0]
            )
            # All edges in order: junction edge + internal edges along the cilium
            cilium_edges = [junction_e] + [
                edge_lookup[(cilium[j], cilium[j + 1])] for j in range(len(cilium) - 1)
            ]
            # BPs between consecutive edges
            cilium_bps = [
                bp_lookup[(cilium_edges[i], cilium_edges[i + 1])]
                for i in range(len(cilium_edges) - 1)
            ]
            per_cilium.append(cilium_bps)

        return np.array(per_cilium, dtype=np.int32)

    def compute_cilia_angles_at_phase(self, phase: float) -> jnp.ndarray:
        """Compute angles for all cilium BPs with metachronal wave offsets.

        Asymmetric power/recovery stroke over one 2π cycle:
          [0,    π/4)  fast power stroke: orthogonal → tangent  (1/8 of cycle)
          [π/4,  π)    hold at tangent                          (3/8 of cycle)
          [π,    2π)   slow recovery: tangent → orthogonal      (1/2 of cycle)

        Returns shape (NUM_CILIA, bps_per_cilium).
        """
        phi = (phase + self.phase_offsets) % (2 * jnp.pi)  # (NUM_CILIA,), in [0, 2π)

        fast_rise = phi / (jnp.pi / 4)                      # 0 → 1 over [0, π/4]
        slow_fall = 1.0 - (phi - jnp.pi) / jnp.pi          # 1 → 0 over [π, 2π]

        stroke = jnp.where(phi < jnp.pi / 4, fast_rise,
                 jnp.where(phi < jnp.pi,     1.0,
                                              slow_fall))
        stroke = jnp.clip(stroke, 0.0, 1.0)  # guard fp rounding at boundaries

        bps_per_cilium = self.per_cilium_bp_indices.shape[1]
        per_segment = self.amplitude * self.stroke_signs * stroke  # (NUM_CILIA,)
        delta = jnp.broadcast_to(per_segment[:, None], (self.NUM_CILIA, bps_per_cilium))

        return self.per_cilium_baseline_angles + delta

    def reset_simulation(self):
        return self.nodes, self.edges, self._make_fields()

    def make_control_fn(self, env: MicrocosmosEnv):
        rotation_speed = self.rotation_speed
        initial_phase = self.initial_phase
        bp_indices = self.per_cilium_bp_indices
        compute = self.compute_cilia_angles_at_phase

        def control_fn(nodes, edges, t):
            phase = t * rotation_speed + initial_phase
            new_cilium_angles = compute(phase)
            new_bending = edges.bending_rest_angles.at[bp_indices.ravel()].set(new_cilium_angles.ravel())
            return edges.__replace__(bending_rest_angles=new_bending)

        return control_fn

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
            new_cilium_angles = self.compute_cilia_angles_at_phase(current_phase)
            new_bending = edges.bending_rest_angles.at[self.per_cilium_bp_indices.ravel()].set(new_cilium_angles.ravel())
            edges = edges.__replace__(bending_rest_angles=new_bending)
            nodes, edges_out, fields = step(nodes, edges, fields, dt, solver_config)
            edges = edges_out
            current_phase += self.rotation_speed
            return nodes, fields

        def extra_info_func(nodes, fields):
            disp = self._update_displacement(nodes)
            return [f"Phase: {current_phase:.2f} rad", f"Displacement: {disp:.4f}"]

        animate_realtime(self.nodes, self.fields, dt, PBD_SCHEME, step_func, extra_info_func,
                         subtitle='cilia')
