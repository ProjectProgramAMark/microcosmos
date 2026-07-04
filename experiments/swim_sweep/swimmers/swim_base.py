from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from microcosmos.simulate import step
from microcosmos.structs.fields import Fields
from microcosmos.rendering import animate, animate_realtime
from microcosmos.solver.config import PBD_SCHEME
from microcosmos.graph import make_edges, compute_bending_pairs
from microcosmos.graph_builder import GraphBuilder
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.utils import displacement


class Swimmer:
    """Shared base for swimming strategy classes."""

    plot_color: str = 'steelblue'
    swimmer_name: str = 'swimmer'

    def __init__(self, cfg: DictConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        viscosity_raw = cfg.experiment.get('viscosity', 0.1)
        viscosity_config = OmegaConf.to_container(viscosity_raw) if OmegaConf.is_config(viscosity_raw) else viscosity_raw
        self.viscosities = [float(v) for v in viscosity_config] if isinstance(viscosity_config, (list, tuple)) else [float(viscosity_config)]

    def run(self):
        if self.cfg.simulation.realtime:
            self.run_realtime()
        else:
            self.run_batch()

    def make_control_fn(self, env):
        """Return control_fn(nodes, edges, t) -> edges, or None for static simulations."""
        return None

    def run_batch(self):
        from microcosmos.env import MicrocosmosEnv
        num_steps = self.cfg.simulation.num_steps
        dt = self.cfg.simulation.dt
        print(f"Running {self.swimmer_name} experiment, viscosities: {self.viscosities}")

        body_length = (self.num_nodes - 1) * self.init_line_distance  # lattice cells
        u_char = PBD_SCHEME.max_fluid_velocity
        print(f"Body length: {body_length:.1f} lattice cells, u_char: {u_char}")
        for viscosity in self.viscosities:
            re = u_char * body_length / viscosity
            print(f"  viscosity={viscosity} -> Re ≈ {re:.1f}")

        self.results_by_viscosity = {}
        for viscosity in self.viscosities:
            solver_config = replace(PBD_SCHEME, viscosity=viscosity)
            env = MicrocosmosEnv(tuple(self.grid_shape), dt, num_steps, solver_config)
            nodes, edges, fields = self.reset_simulation()
            nodes, edges, fields, results = self.run_phase(
                nodes, edges, fields, env,
                f'{self.swimmer_name} (viscosity={viscosity})',
            )
            self.results_by_viscosity[viscosity] = results
        self.save_results()

    def save_results(self):
        if len(self.viscosities) > 1:
            self._save_viscosity_comparison()

        for viscosity, res in self.results_by_viscosity.items():
            has_curvature = 'curvature' in res
            fig, axes = plt.subplots(2 if has_curvature else 1, 1, figsize=(12, 8 if has_curvature else 5))
            if not has_curvature:
                axes = [axes]

            steps = jnp.arange(len(res['displacement']))
            axes[0].plot(steps, res['displacement'], linewidth=2.5, color=self.plot_color)
            axes[0].set_xlabel('Simulation Step')
            axes[0].set_ylabel('Net Displacement')
            axes[0].set_title(f'{self.swimmer_name.title()} Displacement (viscosity={viscosity})')
            axes[0].grid(True, alpha=0.3)

            if has_curvature:
                axes[1].plot(steps, res['curvature'], linewidth=2, color=self.plot_color)
                axes[1].set_xlabel('Simulation Step')
                axes[1].set_ylabel('Hinge Curvature')
                axes[1].set_title('Motion Protocol')
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / f'{self.swimmer_name}_{viscosity}.png', dpi=150)
            plt.close()

            if res.get('trajectory'):
                self._save_animation(res['trajectory'], res['fields_trajectory'],
                                     f"{self.swimmer_name}_{viscosity}", subsample=2)

    def _save_viscosity_comparison(self):
        viscosities = list(self.results_by_viscosity.keys())
        finals = [res['displacement'][-1] for res in self.results_by_viscosity.values()]
        plt.figure(figsize=(10, 6))
        plt.plot(viscosities, finals, 'o-', color=self.plot_color, linewidth=2)
        plt.xlabel('Viscosity')
        plt.ylabel('Total Displacement')
        plt.title(f'{self.swimmer_name.title()} Strategy vs Viscosity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'viscosity_comparison.png')
        plt.close()

    def _make_fields(self):
        h, w = self.grid_shape
        fluid_shape_cfg = self.cfg.experiment.get('fluid_grid_shape', None)
        fh, fw = tuple(fluid_shape_cfg) if fluid_shape_cfg is not None else (h, w)
        weights = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        f_grid = jnp.zeros((9, fh, fw))
        f_grid = f_grid.at[:].set(weights[:, None, None])
        return Fields(
            grid_shape=(h, w),
            steric=jnp.zeros((h, w)),
            fluid_velocity=jnp.zeros((2, fh, fw)),
            f_grid=f_grid,
        )

    def _make_nodes(self, positions, size=None):
        return Nodes(
            position=positions,
            velocity=jnp.zeros_like(positions),
            color_bend=jnp.arange(self.num_nodes) % 3,
            debug_vector=jnp.zeros((self.num_nodes, 2)),
        )

    def _make_edges(self, edge_pairs, theta, num_edges, bending_pairs, bending_rest_angles):
        return Edges(
            pairs=edge_pairs,
            theta=theta,
            rest_lengths=jnp.full(num_edges, self.init_line_distance),
            bending_pairs=bending_pairs,
            bending_rest_angles=bending_rest_angles,
            bending_stiffness=jnp.ones(bending_pairs.shape[0]),
        )

    def _fields_for_render(self, fields):
        """Drop f_grid and downsample fluid_velocity to grid_shape for storage."""
        from dataclasses import replace
        h, w = self.grid_shape
        fH, fW = fields.fluid_velocity.shape[1], fields.fluid_velocity.shape[2]
        fluid_vel = (
            jax.image.resize(fields.fluid_velocity, (2, h, w), method="linear")
            if (fH, fW) != (h, w) else fields.fluid_velocity
        )
        return replace(fields, f_grid=None, fluid_velocity=fluid_vel)

    def _save_animation(self, traj, fields_traj, name, subsample=1, uniform_color=False):
        nodes_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *traj)
        fields_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *fields_traj)
        path = str(self.output_dir / f"{name}.gif")
        animate(nodes_stacked, fields_stacked, filename=path,
                subsample=subsample,
                uniform_color=uniform_color, verbose=True)
        print(f"Animation saved to {path}")

    def _periodic_com(self, positions):
        """Compute center of mass respecting periodic boundaries.

        Uses a reference point (first node) so the mean stays correct
        even when nodes straddle the domain boundary.
        """
        size = tuple(self.grid_shape)
        ref = positions[0]
        deltas = displacement(size, positions, ref)  # (N, 2)
        mean_delta = jnp.mean(deltas, axis=0)
        from microcosmos.utils import periodic_boundary
        return periodic_boundary(size, ref + mean_delta)

    def _init_displacement(self, nodes):
        """Start tracking cumulative wrapped displacement."""
        self._prev_com = self._periodic_com(nodes.position)
        self._cumulative_disp = jnp.zeros(2)
        self._disp_step = 0

    def _check_stability(self, nodes, displacement_history):
        """Raise if NaN or runaway displacement detected."""
        if jnp.any(jnp.isnan(nodes.position)):
            raise RuntimeError("problematic displacement found (NaN)")
        if len(displacement_history) >= 100:
            delta = displacement_history[-1] - displacement_history[-100]
            if delta > 100:
                raise RuntimeError(f"problematic displacement found ({delta:.1f} over 100 steps)")

    def _update_displacement(self, nodes) -> float:
        """Accumulate one step of wrapped CoM displacement, return magnitude.

        If ``displacement_start_step`` is set on the swimmer, displacement
        only accumulates after that many steps (earlier steps return 0).
        """
        current_com = self._periodic_com(nodes.position)
        step_delta = displacement(tuple(self.grid_shape), current_com, self._prev_com)
        self._prev_com = current_com
        self._disp_step += 1
        start = getattr(self, 'displacement_start_step', 0)
        if self._disp_step <= start:
            return 0.0
        self._cumulative_disp = self._cumulative_disp + step_delta
        return float(jnp.linalg.norm(self._cumulative_disp))
