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


class Jellyfish(Swimmer):
    """Bell-pulse jet propulsion, jellyfish-style.

    A ~210° dome arc forms the bell, oriented with the opening facing
    downward (mouth/velum) and the top of the bell facing upward.
    During the power stroke all edges contract and the arc curvature
    tightens (bell closes), squirting water out the opening and
    propelling the bell upward. A slow recovery stroke re-opens the bell.

    Key difference from Squirt: Squirt uses a 330° nearly-closed loop
    (nozzle-jet), whereas Jellyfish uses a wide-open 210° dome that acts
    more like a displacement pump/bell-piston.
    """

    plot_color = 'mediumaquamarine'
    swimmer_name = 'jellyfish'

    def setup(self):
        self.num_nodes = self.cfg.experiment.num_nodes
        self.grid_shape = tuple(self.cfg.experiment.grid_shape)
        self.init_line_distance = self.cfg.experiment.get('init_line_distance', 2.0)
        self.frequency = self.cfg.experiment.get('frequency', 0.05)
        self.rest_length_mod_mag = self.cfg.experiment.get('rest_length_mod_mag', 0.0)
        self.enable_animations = self.cfg.experiment.get('enable_animations', False)
        self.curvature_boost = self.cfg.experiment.get('jellyfish_curvature_boost', 0.0375)
        # Fraction of cycle spent contracting (excluding hold); 0.5 = symmetric.
        # Values < 0.5 → fast contraction / slow recovery (jellyfish-like).
        self.contraction_asymmetry = self.cfg.experiment.get('jellyfish_contraction_asymmetry', 0.15)

        N = self.num_nodes
        d = self.init_line_distance

        edge_pairs, edge_departure_angles = make_edges(N, graph="line", num_instances=1)
        num_edges = edge_pairs.shape[0]

        domain_height, domain_width = self.grid_shape
        size = tuple(self.grid_shape)

        # 210° dome arc — wider open bell than squirt (330°).
        # With initial_dir = π/2 + total_arc/2 and clockwise turns,
        # the midpoint of the arc points straight up (90°), so the dome
        # top faces upward and the opening faces downward.
        total_arc = 7 * jnp.pi / 6   # 210°
        n_bp = N - 2
        bell_turn = -total_arc / max(n_bp, 1)   # clockwise, same convention as Squirt

        bending_rest_angles_shape = jnp.full(n_bp, bell_turn)

        # initial_dir = 90° + 105° = 195°  (first edge points lower-left)
        initial_dir = jnp.pi / 2 + total_arc / 2
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

        self.base_bending_rest_angles = bending_rest_angles
        self.num_edges = num_edges

        self.nodes = self._make_nodes(positions, size)
        self.edges = self._make_edges(
            edge_pairs, edge_theta, num_edges,
            bending_pairs, bending_rest_angles,
        )
        self.fields = self._make_fields()

    def _stroke_envelope(self, t: float) -> float:
        """Contraction → brief hold → recovery. Returns value in [0, 1].

        contraction_asymmetry controls the split of non-hold time:
          0.5 → equal contraction and recovery (symmetric)
          <0.5 → fast contraction / slow recovery (jellyfish-like)
        """
        period = 1.0 / self.frequency
        phase = (t % period) / period

        hold_frac = 0.05
        fast_frac = self.contraction_asymmetry * (1.0 - hold_frac)

        if phase < fast_frac:
            s = phase / fast_frac
        elif phase < fast_frac + hold_frac:
            s = 1.0
        else:
            s = 1.0 - (phase - fast_frac - hold_frac) / (1.0 - fast_frac - hold_frac)

        return float(jnp.clip(s, 0.0, 1.0))

    def compute_rest_lengths_at_time(self, t: float) -> jnp.ndarray:
        s = self._stroke_envelope(t)
        scale = 1.0 - self.rest_length_mod_mag * s
        return jnp.full(self.num_edges, self.init_line_distance * scale)

    def compute_bending_rest_angles_at_time(self, t: float) -> jnp.ndarray:
        s = self._stroke_envelope(t)
        # Tighten the arc during contraction (more negative → tighter dome)
        return self.base_bending_rest_angles - self.curvature_boost * s

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
        viscosity = self.cfg.experiment.get('realtime_viscosity', 0.1)
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
                         subtitle="Jellyfish")
