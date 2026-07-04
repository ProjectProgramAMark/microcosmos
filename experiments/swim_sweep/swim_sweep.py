from dataclasses import replace
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from microcosmos.solver.config import PBD_SCHEME, COSSERAT_SCHEME
from microcosmos.rendering import animate, render_frame
from microcosmos.structs.fields import Fields
from microcosmos.env import MicrocosmosEnv

from ..base_experiment import Experiment
from .swimmers.ray import Ray
from .swimmers.worm import Worm
from .swimmers.tadpole import Tadpole
from .swimmers.squirt import Squirt
from .swimmers.cilia import Cilia
from .swimmers.jellyfish import Jellyfish
from .swimmers.fast_worm import FastWorm
from .swimmers.turbo_worm import TurboWorm

SWIMMERS = {
    "ray": Ray,
    "worm": Worm,
    "tadpole": Tadpole,
    "squirt": Squirt,
    "cilia": Cilia,
    "jellyfish": Jellyfish,
    "fast_worm": FastWorm,
    "turbo_worm": TurboWorm,
}


class SwimSweepExperiment(Experiment):
    """Compare swimming strategies across a sweep of viscosities.

    Runs the configured swimmers at each viscosity and plots total
    displacement vs viscosity for each.
    """

    def setup(self):
        viscosity_config = OmegaConf.to_container(self.cfg.experiment.viscosities)
        self.viscosities = [float(v) for v in viscosity_config]
        self.num_steps = self.cfg.simulation.num_steps
        self.dt = self.cfg.simulation.dt
        self.enable_animations = self.cfg.experiment.get("enable_animations", False)
        self.subsample = self.cfg.experiment.get("animation_subsample", 4)
        self.displacement_start_step = self.cfg.experiment.get("displacement_start_step", 0)
        self.snapshot_step = self.cfg.experiment.get("snapshot_step", 150)
        # Build base solver scheme; list-valued params become sweep axes
        scheme_name = str(self.cfg.experiment.get('scheme', 'pbd'))
        base = COSSERAT_SCHEME if scheme_name == 'stable_cosserat' else PBD_SCHEME

        SOLVER_PARAMS = ['damping', 'max_fluid_velocity', 'synthetic_node_width', 'ibm_kernel_size']
        sweep_params = {}
        base_overrides = {}
        for key in SOLVER_PARAMS:
            val = self.cfg.experiment.get(key, None)
            if val is None:
                continue
            container = OmegaConf.to_container(val) if OmegaConf.is_config(val) else val
            if isinstance(container, list):
                sweep_params[key] = [float(v) for v in container]
            else:
                base_overrides[key] = float(container)

        # Scheme-specific stiffness keys so you can keep both tuned simultaneously
        stretch_key = f'stiffness_stretch_{scheme_name}' if scheme_name != 'pbd' else 'stiffness_stretch_pbd'
        shear_key   = f'stiffness_shear_{scheme_name}'   if scheme_name != 'pbd' else 'stiffness_shear_pbd'
        if self.cfg.experiment.get(stretch_key) is not None:
            base_overrides['stiffness_stretch'] = float(self.cfg.experiment[stretch_key])
        if self.cfg.experiment.get(shear_key) is not None:
            base_overrides['stiffness_shear'] = float(self.cfg.experiment[shear_key])
        if scheme_name == 'stable_cosserat':
            base_overrides['dt'] = self.dt  # w = 1/dt² must match simulation dt

        self.base_scheme = base.__replace__(**base_overrides)
        self.pbd_scheme = self.base_scheme  # alias kept for any downstream references

        if sweep_params:
            from itertools import product as cartesian
            keys = list(sweep_params.keys())
            values = [sweep_params[k] for k in keys]
            self.sweep_conditions = []
            for combo in cartesian(*values):
                overrides = dict(zip(keys, combo))
                label = ", ".join(f"{k}={v}" for k, v in overrides.items())
                self.sweep_conditions.append((label, overrides))
        else:
            self.sweep_conditions = [("default", {})]
        print(f"Sweep conditions: {[label for label, _ in self.sweep_conditions]}")

        swimmer_names = OmegaConf.to_container(self.cfg.experiment.swimmers)
        self.swimmers = {}
        for name in swimmer_names:
            if name not in SWIMMERS:
                raise ValueError(f"Unknown swimmer '{name}'. Choose from: {list(SWIMMERS)}")
            print(f"Setting up {name}...")
            swimmer = SWIMMERS[name](self.cfg, self.output_dir)
            swimmer.setup()
            self.swimmers[name] = swimmer

    def loss_fn(self, params):
        raise NotImplementedError("SwimSweepExperiment does not use optimization")

    def _save_animation(self, traj, fields_traj, name: str):
        nodes_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *traj)
        fields_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *fields_traj)
        path = str(self.output_dir / f"{name}.gif")
        animate(nodes_stacked, fields_stacked, filename=path,
                subsample=self.subsample, verbose=False)
        print(f"    Saved {path}")

    def _run_swimmer(self, swimmer, viscosity: float, label: str, capture_trajectory: bool = False, base_scheme=None):
        solver_config = replace(base_scheme or self.base_scheme, viscosity=viscosity)
        env = MicrocosmosEnv(
            grid_shape=tuple(swimmer.grid_shape),
            dt=self.dt,
            num_steps=self.num_steps,
            solver_config=solver_config,
        )
        swimmer.displacement_start_step = self.displacement_start_step
        nodes, edges, fields = swimmer.reset_simulation()
        orig = swimmer.enable_animations
        swimmer.enable_animations = self.enable_animations or capture_trajectory
        _, _, _, results = swimmer.run_phase(nodes, edges, fields, env, label)
        swimmer.enable_animations = orig
        displacement = results['displacement'][-1] if results['displacement'] else 0.0
        return displacement, results['trajectory'], results['fields_trajectory']

    def run(self):
        all_results = {}  # (sweep_label, swimmer_name) -> [disp_per_viscosity]
        snapshots = {}

        first_swimmer = next(iter(self.swimmers.values()))
        self.body_length = (first_swimmer.num_nodes - 1) * first_swimmer.init_line_distance
        self.u_char = self.base_scheme.max_fluid_velocity
        print(f"\nBody length: {self.body_length:.1f} lattice cells, u_char: {self.u_char}")
        for viscosity in self.viscosities:
            re = self.u_char * self.body_length / viscosity
            print(f"  viscosity={viscosity} -> Re ≈ {re:.1f}")

        snapshot_vis_idx = len(self.viscosities) // 2

        for sweep_label, overrides in self.sweep_conditions:
            if sweep_label != "default":
                print(f"\n=== Sweep: {sweep_label} ===")
            scheme = self.base_scheme.__replace__(**overrides)

            for vis_idx, viscosity in enumerate(self.viscosities):
                print(f"\n--- Viscosity = {viscosity} ---")
                v_tag = str(viscosity).replace(".", "_")
                capture = (vis_idx == snapshot_vis_idx) and (sweep_label == self.sweep_conditions[0][0])

                for name, swimmer in self.swimmers.items():
                    print(f"  {name}...", end=" ", flush=True)
                    d, traj, fields_traj = self._run_swimmer(
                        swimmer, viscosity, f"  {name}",
                        capture_trajectory=capture, base_scheme=scheme,
                    )
                    key = (sweep_label, name)
                    if key not in all_results:
                        all_results[key] = []
                    all_results[key].append(d)
                    print(f"{d:.4f}")
                    if capture and traj:
                        nodes_ts = jax.tree.map(lambda *xs: jnp.stack(xs), *traj)
                        fields_ts = Fields(
                            grid_shape=fields_traj[0].grid_shape,
                            steric=jnp.stack([f.steric for f in fields_traj]),
                            fluid_velocity=jnp.stack([f.fluid_velocity for f in fields_traj]),
                            f_grid=None,
                            energy=None,
                        )
                        snapshots[name] = (nodes_ts, fields_ts)
                    if self.enable_animations and traj:
                        sweep_tag = sweep_label.replace(", ", "_").replace("=", "") if sweep_label != "default" else ""
                        anim_name = f"{name}_v{v_tag}_{sweep_tag}" if sweep_tag else f"{name}_v{v_tag}"
                        self._save_animation(traj, fields_traj, anim_name)

        self.all_results = all_results
        self.results = {name: all_results.get((self.sweep_conditions[0][0], name), [])
                        for name in self.swimmers}
        self.snapshots = snapshots
        self.save_results()

    def _save_sweep_bar_plot(self):
        v = self.viscosities
        swimmer_names = list(self.swimmers.keys())
        n_swimmers = len(swimmer_names)
        n_sweeps = len(self.sweep_conditions)

        sweep_cmap = plt.cm.get_cmap("tab10", max(n_sweeps, 2))
        markers = ["o", "s", "D", "^", "v", "P", "*", "X"]

        fig, axes = plt.subplots(n_swimmers, 1, figsize=(10, 4 * n_swimmers),
                                 sharex=True, squeeze=False)

        for m_idx, name in enumerate(swimmer_names):
            ax = axes[m_idx, 0]
            for s_idx, (sweep_label, _) in enumerate(self.sweep_conditions):
                disps = self.all_results.get((sweep_label, name), [0] * len(v))
                marker = markers[s_idx % len(markers)]
                ax.plot(v, disps, f"{marker}-", label=sweep_label,
                        color=sweep_cmap(s_idx), linewidth=2, markersize=7)

            ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            ax.set_xscale("log")
            ax.set_ylabel("Displacement [lu]", fontsize=12)
            ax.set_title(name.capitalize(), fontsize=14)
            ax.legend(fontsize=10)
            ax.tick_params(axis='both', labelsize=11)
            ax.grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel("Viscosity", fontsize=14)
        fig.suptitle("Swimming Strategy Sweep Comparison", fontsize=15, y=1.01)
        plt.tight_layout()

        out = self.output_dir / "sweep_comparison.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        repo_root = Path(__file__).parents[2]
        paper_out = repo_root / "paper" / "figures" / "figure2.pdf"
        plt.savefig(paper_out, bbox_inches='tight')
        plt.close()
        print(f"\nSaved to {out}")
        print(f"Saved to {paper_out}")

    @staticmethod
    def _crop_to_creature(img, positions, grid_shape, window_fraction=0.5):
        """Crop img (H, W, 3) around creature centroid and resize back to original size.

        positions: (N, 2) lattice-unit node positions at snapshot frame
        grid_shape: (h, w) domain size in lattice units (from Fields.grid_shape)
        window_fraction: side of crop window as a fraction of the domain (0 < f <= 1)
        """
        h_img, w_img = img.shape[:2]
        h_lu, w_lu = grid_shape

        cx_lu = float(np.mean(positions[:, 0]))
        cy_lu = float(np.mean(positions[:, 1]))

        cx_px = cx_lu / w_lu * w_img
        cy_px = cy_lu / h_lu * h_img

        half_w = int(w_img * window_fraction / 2)
        half_h = int(h_img * window_fraction / 2)

        x0 = int(np.clip(cx_px - half_w, 0, w_img))
        x1 = int(np.clip(cx_px + half_w, 0, w_img))
        y0 = int(np.clip(cy_px - half_h, 0, h_img))
        y1 = int(np.clip(cy_px + half_h, 0, h_img))

        crop = img[y0:y1, x0:x1]
        return cv2.resize(crop, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    def save_results(self):
        if len(self.sweep_conditions) > 1:
            self._save_sweep_bar_plot()
            return
        v = self.viscosities
        n_swimmers = len(self.swimmers)
        snapshot_step = self.snapshot_step
        snapshots = getattr(self, 'snapshots', {})

        # Render swimmer thumbnails if trajectories were captured
        thumb_imgs = {}
        for name, (nodes_ts, fields_ts) in snapshots.items():
            frame_idx = min(int(snapshot_step * 1.2), nodes_ts.position.shape[0] - 1)
            # frame_idx = min(int(snapshot_step), nodes_ts.position.shape[0] - 1)
            img = render_frame(nodes_ts, fields_ts, frame_idx, sz=512)
            positions = np.array(nodes_ts.position[frame_idx])
            img = self._crop_to_creature(img, positions, fields_ts.grid_shape, window_fraction=0.35)
            thumb_imgs[name] = img

        has_thumbs = bool(thumb_imgs)
        if has_thumbs:
            fig = plt.figure(figsize=(10, 12))
            gs = fig.add_gridspec(n_swimmers, 2, width_ratios=[1, 2], hspace=0.4, wspace=0.08)
            ax = fig.add_subplot(gs[:, 1])
            for i, name in enumerate(self.swimmers):
                ax_img = fig.add_subplot(gs[i, 0])
                if name in thumb_imgs:
                    ax_img.imshow(thumb_imgs[name])
                ax_img.set_title(name.capitalize(), fontsize=14, pad=4)
                ax_img.axis('off')
        else:
            fig, ax = plt.subplots(figsize=(6, 12))

        markers = ["^", "D", "P", "o", "s", "*", "v", "X", "d", "h"]
        cmap = plt.cm.get_cmap("tab10" if n_swimmers <= 10 else "tab20", n_swimmers)
        for i, (name, _swimmer) in enumerate(self.swimmers.items()):
            marker = markers[i % len(markers)]
            ax.plot(v, self.results[name], f"{marker}-",
                    label=name.capitalize(), color=cmap(i), linewidth=2)

        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("Viscosity", fontsize=14)
        ax.set_ylabel("Total Displacement [lu]", fontsize=14)
        ax.set_title("Swimming Strategy Comparison vs Viscosity", fontsize=15)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        out = self.output_dir / "sweep_comparison.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"\nSaved to {out}")
