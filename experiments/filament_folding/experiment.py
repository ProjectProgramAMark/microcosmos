import io
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import optax
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import jax
import jax.numpy as jnp

_FONT_SEARCH_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _find_system_font():
    for path in _FONT_SEARCH_PATHS:
        if Path(path).exists():
            return path
    return None


def _load_pil_font(font_path, render_size):
    from PIL import ImageFont
    font_size = int(render_size * 1.0)
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, font_size)
    auto = _find_system_font()
    if auto:
        return ImageFont.truetype(auto, font_size)
    return ImageFont.load_default()

from hydra.utils import get_original_cwd
from experiments.base_experiment import Experiment
from experiments.filament_folding.utils import init_nodes, downsample_particles_to_grid
from microcosmos.rendering import animate
from microcosmos.simulate import step as sim_step, simulate
from microcosmos.structs.fields import Fields
from microcosmos.solver.config import PBD_SCHEME_NO_FLUID, COSSERAT_SCHEME_NO_FLUID
from microcosmos.utils import displacement

_SCHEMES = {
    "pbd": PBD_SCHEME_NO_FLUID,
    "stable_cosserat": COSSERAT_SCHEME_NO_FLUID,
}


def _chamfer_loss(particles: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Bidirectional Chamfer distance between particle positions and target point cloud."""
    diff = particles[:, None, :] - targets[None, :, :]  # (N, M, 2)
    dists2 = jnp.sum(diff ** 2, axis=-1)  # (N, M)
    nn_particles = jnp.min(dists2, axis=1)  # (N,) — each particle to nearest target
    nn_targets = jnp.min(dists2, axis=0)    # (M,) — each target to nearest particle
    return jnp.mean(nn_particles) + jnp.mean(nn_targets)


def _pca_rotation_angle(particles: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """PCA-based rotation angle to align particles to target orientation.

    Returns the base angle only. Callers handle the 180° ambiguity by computing
    losses for both base_angle and base_angle+π and taking jnp.minimum — no
    stop_gradient or discrete flip selection needed.
    """
    p_centered = particles - jnp.mean(particles, axis=0)
    t_centered = targets - jnp.mean(targets, axis=0)

    eps = 1e-6 * jnp.eye(2)
    cov_p = (p_centered.T @ p_centered) / p_centered.shape[0] + eps
    cov_t = (t_centered.T @ t_centered) / t_centered.shape[0] + eps
    _, evecs_p = jnp.linalg.eigh(cov_p)
    _, evecs_t = jnp.linalg.eigh(cov_t)
    v_p = evecs_p[:, -1]
    v_t = evecs_t[:, -1]

    return jnp.arctan2(v_t[1], v_t[0]) - jnp.arctan2(v_p[1], v_p[0])


def _rotate_particles(particles: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    """Rotate centered particle cloud by angle."""
    p_centered = particles - jnp.mean(particles, axis=0)
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack([p_centered[:, 0] * c - p_centered[:, 1] * s,
                      p_centered[:, 0] * s + p_centered[:, 1] * c], axis=-1)


def _chamfer_loss_pca_aligned(
    particles: jnp.ndarray, targets: jnp.ndarray
) -> jnp.ndarray:
    """Chamfer loss after PCA alignment. Both ±180° orientations computed; jnp.minimum taken."""
    angle = jax.lax.stop_gradient(_pca_rotation_angle(particles, targets))
    t_centered = targets - jnp.mean(targets, axis=0)
    loss_0 = _chamfer_loss(_rotate_particles(particles, angle), t_centered)
    loss_pi = _chamfer_loss(_rotate_particles(particles, angle + jnp.pi), t_centered)
    return jnp.minimum(loss_0, loss_pi)


def _dcd_loss(particles: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Density-Aware Chamfer Distance.

    Same O(N×M) cost as standard chamfer but down-weights particles that cluster
    on the same target point: contribution of particle i is divided by the number
    of other particles that also map to the same nearest target. Prevents the
    mode-collapse exploit where many nodes pile on one dense region.
    """
    N = particles.shape[0]
    M = targets.shape[0]
    diff = particles[:, None, :] - targets[None, :, :]  # (N, M, 2)
    dists2 = jnp.sum(diff ** 2, axis=-1)  # (N, M)

    # particles → targets
    nn_p = jnp.argmin(dists2, axis=1)   # (N,) index of nearest target per particle
    d_p  = jnp.min(dists2, axis=1)      # (N,) squared distance
    counts_p = jax.lax.stop_gradient(jnp.zeros(M).at[nn_p].add(1.0))  # (M,) density counts
    loss_pt = jnp.mean(d_p / counts_p[nn_p])

    # targets → particles
    nn_t = jnp.argmin(dists2, axis=0)   # (M,)
    d_t  = jnp.min(dists2, axis=0)      # (M,)
    counts_t = jax.lax.stop_gradient(jnp.zeros(N).at[nn_t].add(1.0))  # (N,)
    loss_tp = jnp.mean(d_t / counts_t[nn_t])

    return loss_pt + loss_tp


def _dcd_loss_pca_aligned(particles: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """DCD loss after PCA alignment. Both ±180° orientations; jnp.minimum taken."""
    angle = jax.lax.stop_gradient(_pca_rotation_angle(particles, targets))
    t_centered = targets - jnp.mean(targets, axis=0)
    loss_0  = _dcd_loss(_rotate_particles(particles, angle), t_centered)
    loss_pi = _dcd_loss(_rotate_particles(particles, angle + jnp.pi), t_centered)
    return jnp.minimum(loss_0, loss_pi)


def _smooth_max_nn(particles: jnp.ndarray, targets: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """Smooth maximum of p→t nearest-neighbor distances via LogSumExp.

    Returns logsumexp(alpha * d_i) / alpha, which approximates max(d_i) for
    large alpha and penalises outlier particles that sit far from any target.
    """
    diff = particles[:, None, :] - targets[None, :, :]   # (N, M, 2)
    dists2 = jnp.sum(diff ** 2, axis=-1)                  # (N, M)
    nn_dists = jnp.sqrt(jnp.min(dists2, axis=1) + 1e-8)   # (N,) L2 distances
    return jax.scipy.special.logsumexp(alpha * nn_dists) / alpha


def _smooth_max_nn_pca_aligned(
    particles: jnp.ndarray, targets: jnp.ndarray, alpha: float = 0.1
) -> jnp.ndarray:
    """Smooth-max nn loss after PCA alignment. Both ±180° orientations; jnp.minimum taken."""
    angle = jax.lax.stop_gradient(_pca_rotation_angle(particles, targets))
    t_centered = targets - jnp.mean(targets, axis=0)
    sm_0  = _smooth_max_nn(_rotate_particles(particles, angle),            t_centered, alpha)
    sm_pi = _smooth_max_nn(_rotate_particles(particles, angle + jnp.pi),  t_centered, alpha)
    return jnp.minimum(sm_0, sm_pi)


def _feedback_mlp_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """3→hidden→1 MLP with tanh activations. Returns scalar correction."""
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    return jnp.tanh(jnp.dot(h, params["W2"]) + params["b2"])


def _compute_feedback_corrections(
    mlp_params: dict,
    edges,
    target_bending_angles: jnp.ndarray,
    correction_scale: float,
) -> jnp.ndarray:
    """Per-bending-pair additive correction based on local curvature error.

    Input to MLP: [error[i-1], error[i], error[i+1]] (neighbor errors, padded at ends).
    Output: correction in [-correction_scale, +correction_scale].
    At equilibrium (error=0) the MLP output is 0, so no extra force is applied.
    """
    e_in = edges.bending_pairs[:, 0]
    e_out = edges.bending_pairs[:, 1]
    current_bending = edges.theta[e_out] - edges.theta[e_in]
    error = (current_bending - target_bending_angles + jnp.pi) % (2 * jnp.pi) - jnp.pi  # (B,)

    error_left = jnp.concatenate([error[:1], error[:-1]])
    error_right = jnp.concatenate([error[1:], error[-1:]])
    inputs = jnp.stack([error_left, error, error_right], axis=-1)  # (B, 3)

    raw = jax.vmap(lambda x: _feedback_mlp_forward(mlp_params, x))(inputs)  # (B,)
    return raw * correction_scale


def _load_png_chw(path: Path) -> np.ndarray:
    """Read a PNG from disk and return (C, H, W) float32 in [0, 1]."""
    img = plt.imread(str(path))  # (H, W) or (H, W, C[A])
    if img.ndim == 2:
        img = img[None]  # (1, H, W)
    else:
        img = img[:, :, :3].transpose(2, 0, 1)  # (3, H, W)
    return img.astype(np.float32)


class FilamentFoldingExperiment(Experiment):
    """Gradient-based optimization of a line filament toward an MNIST digit target."""

    def setup(self):
        characters = list(self.cfg.experiment.get("characters", []))
        if len(characters) > 1:
            self._multi_char = True
            self.characters = characters
            self.char_exps: dict[str, "FilamentFoldingExperiment"] = {}
            for char in characters:
                char_cfg = OmegaConf.merge(
                    self.cfg,
                    OmegaConf.create({"experiment": {"character": char, "characters": [char]}}),
                )
                char_dir = self.output_dir / char.lower()
                exp = FilamentFoldingExperiment(char_cfg, char_dir)
                exp.setup()
                self.char_exps[char] = exp
            self.tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
            self.loss_history = []
            self.shape_loss_history = []
            return
        self._multi_char = False

        cfg = self.cfg.experiment
        init_cfg = cfg.initialization
        loss_cfg = cfg.loss
        grid_shape = tuple(cfg.grid_shape)
        num_nodes = int(cfg.num_nodes)
        h, w = grid_shape

        self.nodes, self.edges = init_nodes(
            num_nodes, grid_shape, initialization="line",
            init_line_distance=init_cfg.init_line_distance,
        )

        self.overfit_single_init = bool(init_cfg.get("overfit_single_init", False))
        if self.overfit_single_init:
            self.batch_size = 1
            self.overfit_key = jax.random.PRNGKey(int(init_cfg.get("overfit_seed", 0)))
        else:
            self.batch_size = int(self.cfg.optimization.get("batch_size", 1))
        self.key = jax.random.PRNGKey(42)
        mid = num_nodes // 2
        self.pin_middle_nodes = bool(init_cfg.get("pin_middle_nodes", True))
        self.pin_indices = jnp.array([mid, mid + 1])
        self.pin_edge_idx = mid  # edge mid connects node mid→mid+1; anchors global orientation

        self.target_digit = int(cfg.digit)
        self.target_font = str(cfg.get("font", None) or "").strip() or None
        self.target_character = str(cfg.get("character", "A")).strip()
        _raw_image_path = str(cfg.get("image_path", None) or "").strip() or None
        if _raw_image_path:
            p = Path(_raw_image_path)
            if not p.is_absolute():
                p = Path(get_original_cwd()) / p
            self.target_image_path = p
        else:
            self.target_image_path = None

        if self.target_image_path:
            seed_key = jax.random.PRNGKey(hash(self.target_image_path.stem) & 0x7FFFFFFF)
        elif self.target_font:
            seed_key = jax.random.PRNGKey(ord(self.target_character[0]))
        else:
            seed_key = jax.random.PRNGKey(int(cfg.digit))
        self.nodes, self.edges = self._make_init_state(seed_key)

        weights = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.fields = Fields(
            grid_shape=grid_shape,
            steric=jnp.zeros(grid_shape),
            fluid_velocity=jnp.zeros((2, h, w)),
            f_grid=jnp.zeros((9, h, w)).at[:].set(weights[:, None, None]),
        )
        self.target_size = loss_cfg.target_size
        self.curvature_scale = loss_cfg.curvature_scale
        self.distance_min = loss_cfg.distance_min
        self.distance_max = loss_cfg.distance_max
        self.smoothness_weight = float(loss_cfg.get("smoothness_weight", 0.0))
        self.max_dist_weight = float(loss_cfg.get("max_dist_weight", 0.0))
        self.smooth_max_alpha = float(loss_cfg.get("smooth_max_alpha", 0.1))
        self.use_pca_rotation = bool(loss_cfg.get("use_pca_rotation", True))
        self.num_stability_frames = int(loss_cfg.get("num_stability_frames", 1))
        self.stability_anchor_offset = int(loss_cfg.get("stability_anchor_offset", 30))

        self.downsample_particles_to_grid = partial(
            downsample_particles_to_grid,
            target_size=self.target_size,
            sdf_cap=loss_cfg.sdf_cap,
            grid_shape=grid_shape,
        )

        self.loss_mode = str(loss_cfg.get("loss_mode", "sdf"))
        self.pointcloud_num_points = int(loss_cfg.get("pointcloud_num_points", 400))
        self.num_eval_frames = int(loss_cfg.get("num_eval_frames", 3))
        if self.target_image_path:
            self.target_image = self._load_image_target()
        elif self.target_font:
            self.target_image = self._load_font_target()
        else:
            self.target_image = self._load_mnist_target()
        self._save_density_png(self.target_image, self.output_dir / "target_image.png")

        if self.loss_mode in ("pointcloud", "hybrid", "dcd"):
            self.target_pointcloud = self._load_target_pointcloud(self.pointcloud_num_points)
            self._save_pointcloud_png(self.target_pointcloud, self.output_dir / "target_pointcloud.png")

        self.scheme = self._build_scheme()
        phys = self.cfg.get("physics", {})
        bending_stiffness_val = float(phys.get("bending_stiffness", 0.5))
        if bending_stiffness_val != 0.5:
            from dataclasses import replace as dc_replace
            self.edges = dc_replace(self.edges, bending_stiffness=jnp.full_like(self.edges.bending_stiffness, bending_stiffness_val))

        from omegaconf import OmegaConf
        curriculum_cfg = self.cfg.experiment.get("curriculum", None)
        self.curriculum_stages = []
        if curriculum_cfg is not None:
            stages = []
            for s in curriculum_cfg.values():
                raw = OmegaConf.to_container(s.params, resolve=True)
                flat = {}
                def _flatten(d, prefix=""):
                    for k, v in d.items():
                        key = f"{prefix}.{k}" if prefix else k
                        _flatten(v, key) if isinstance(v, dict) else flat.__setitem__(key, v)
                _flatten(raw)
                stages.append((int(s.start), flat))
            self.curriculum_stages = sorted(stages, key=lambda x: x[0])
            print(f"[curriculum] {len(self.curriculum_stages)} stages: " +
                  ", ".join(f"step {s} → {p}" for s, p in self.curriculum_stages))

        opt_cfg = self.cfg.optimization
        lr = float(opt_cfg.learning_rate)
        lr_min = float(opt_cfg.get("lr_min", 1e-3))
        warmup_steps = int(opt_cfg.get("warmup_steps", 0))
        total_steps = int(opt_cfg.num_steps)
        grad_clip = float(opt_cfg.get("grad_clip", float("inf")))

        # Global cosine — used only as a reference to compute each stage's LR floor.
        _global_cosine = optax.cosine_decay_schedule(
            lr, max(total_steps - warmup_steps, 1), alpha=lr_min / lr
        )

        def _global_lr_at(step: int) -> float:
            return float(_global_cosine(max(step - warmup_steps, 0)))

        if self.curriculum_stages:
            starts = [s for s, _ in self.curriculum_stages]
            ends = starts[1:] + [total_steps]

            per_stage = []
            for i, (stage_start, stage_end) in enumerate(zip(starts, ends)):
                # NOTE: join_schedules passes (step - boundary) as local step to each
                # sub-schedule, so all schedules below see steps starting from 0.
                stage_len = max(stage_end - stage_start, 1)
                end_lr = max(_global_lr_at(stage_end), 1e-8)

                if i == 0 and warmup_steps > 0:
                    # First stage: linear warmup (local 0→wu) then exponential decay.
                    wu = min(warmup_steps, stage_len)
                    exp_len = max(stage_len - wu, 1)
                    # Inner join_schedules also offsets: exp part sees local steps 0..exp_len.
                    per_stage.append(optax.join_schedules(
                        schedules=[
                            optax.linear_schedule(lr * 0.01, lr, wu),
                            optax.exponential_decay(lr, transition_steps=exp_len, decay_rate=end_lr / lr, end_value=end_lr),
                        ],
                        boundaries=[wu],
                    ))
                else:
                    # Each stage resets to peak lr at local step 0 and decays
                    # exponentially to the global cosine floor at local step stage_len.
                    per_stage.append(
                        optax.exponential_decay(lr, transition_steps=stage_len, decay_rate=end_lr / lr, end_value=end_lr)
                    )

            schedule = optax.join_schedules(schedules=per_stage, boundaries=starts[1:])
        else:
            schedule = optax.join_schedules(
                schedules=[
                    optax.linear_schedule(lr * 0.01, lr, warmup_steps),
                    optax.cosine_decay_schedule(lr, total_steps - warmup_steps, alpha=lr_min / lr),
                ],
                boundaries=[warmup_steps],
            )

        self._lr_schedule = schedule
        self._total_steps = total_steps
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adam(schedule),
        )
        self.params = {
            "bending_rest_angles_raw": jnp.zeros_like(self.edges.bending_rest_angles),
            "rest_lengths_raw": jnp.zeros_like(self.edges.rest_lengths),
        }

        fb_cfg = self.cfg.experiment.get("feedback_control", {})
        self.use_feedback = bool(fb_cfg.get("enabled", False))
        if self.use_feedback:
            hidden_dim = int(fb_cfg.get("hidden_dim", 8))
            self.feedback_correction_scale = float(fb_cfg.get("correction_scale", self.curvature_scale))
            key_mlp = jax.random.PRNGKey(99)
            k1, k2 = jax.random.split(key_mlp)
            self.params["feedback_mlp"] = {
                "W1": jax.random.normal(k1, (3, hidden_dim)) * 0.01,
                "b1": jnp.zeros(hidden_dim),
                "W2": jnp.zeros(hidden_dim),
                "b2": jnp.zeros(()),
            }
            print(f"[feedback_control] enabled: hidden_dim={hidden_dim}, correction_scale={self.feedback_correction_scale}")

        self.opt_state = self.optimizer.init(self.params)
        self.loss_history = []
        self.shape_loss_history = []
        self.curriculum_events = []  # list of (step, params_dict) for loss plot markers
        self.viz_every = int(self.cfg.optimization.get("viz_every", 0))

        from omegaconf import OmegaConf
        self.tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        self.tb.add_text("config", f"```yaml\n{OmegaConf.to_yaml(self.cfg)}\n```", global_step=0)

    def _save_lr_schedule_plot(self):
        import numpy as np
        steps = np.arange(self._total_steps)
        lrs = np.array([float(self._lr_schedule(int(s))) for s in steps])

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(steps, lrs, color="steelblue", linewidth=1.5)

        for start, _ in self.curriculum_stages:
            ax.axvline(x=start, color="red", linestyle="--", alpha=0.6, linewidth=1.0)

        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule (bespoke curriculum)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "lr_schedule.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[lr_schedule] saved plot → {self.output_dir / 'lr_schedule.png'}")
        self.tb.add_image("schedule/lr", _load_png_chw(self.output_dir / "lr_schedule.png"), global_step=0)

    def _make_init_state(self, key, noise=None):
        """Dispatch to the configured init mode. noise overrides fourier_phase_noise if given."""
        init_mode = str(self.cfg.experiment.initialization.get("init_mode", "random_walk"))
        if init_mode == "fourier_series":
            nodes, edges = self._make_fourier_init_state(key, noise=noise)
        else:
            nodes, edges = self._make_random_walk_init_state(key)
        if self.cfg.experiment.initialization.get("canonical_init", False):
            nodes, edges = self._apply_canonical_init_rotation(nodes, edges)
        return nodes, edges

    def _apply_canonical_init_rotation(self, nodes, edges):
        """Rotate entire init shape so first→last node vector points up (+y)."""
        pos = nodes.position
        center = pos.mean(axis=0)
        pos_c = pos - center
        v = pos_c[-1] - pos_c[0]
        angle = jnp.arctan2(v[1], v[0])
        rot_angle = jnp.pi / 2.0 - angle
        c, s = jnp.cos(rot_angle), jnp.sin(rot_angle)
        rot = jnp.array([[c, -s], [s, c]])
        pos_rotated = pos_c @ rot.T + center
        nodes = nodes.__replace__(position=pos_rotated)
        edges = edges.__replace__(theta=edges.theta + rot_angle)
        return nodes, edges

    def _make_random_walk_init_state(self, key):
        """Random-walk init: rotate so mid-segment points along +x, translate center of mass to grid center."""
        cfg = self.cfg.experiment
        init_cfg = cfg.initialization
        grid_shape = tuple(cfg.grid_shape)
        num_nodes = int(cfg.num_nodes)
        init_line_distance = init_cfg.init_line_distance
        h, w = grid_shape
        mid = num_nodes // 2

        walk_angle_std = float(init_cfg.get("walk_angle_std", 0.3))
        angle_increments = jax.random.normal(key, (num_nodes - 1,)) * walk_angle_std
        angles = jnp.cumsum(angle_increments)
        steps = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1) * init_line_distance
        walk = jnp.concatenate([jnp.zeros((1, 2)), jnp.cumsum(steps, axis=0)], axis=0)

        # Rotate so mid→mid+1 points along +x, then translate so their midpoint is at grid center
        seg = walk[mid + 1] - walk[mid]
        seg_angle = jnp.arctan2(seg[1], seg[0])
        cos_a, sin_a = jnp.cos(-seg_angle), jnp.sin(-seg_angle)
        rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
        walk = (walk - walk[mid]) @ rot.T
        center_of_mass = walk.mean(axis=0)
        center = jnp.array([w / 2.0, h / 2.0])
        position = walk - center_of_mass + center

        nodes = self.nodes.__replace__(position=position)

        edges = self.edges
        if init_cfg.get("theta_pre_compute", False):
            src = edges.pairs[:, 0]
            tgt = edges.pairs[:, 1]
            deltas = displacement(grid_shape, nodes.position[tgt], nodes.position[src])
            edges = edges.__replace__(theta=jnp.arctan2(deltas[:, 1], deltas[:, 0]))

        return nodes, edges

    def _make_fourier_init_state(self, key, noise=None):
        """Fourier-series init: heading angle = sum of low-freq sinusoids.

        Produces a smooth compact serpentine. Phases are randomized from key
        (unless fourier_randomize_phases: false, in which case fourier_phases is used).
        noise overrides fourier_phase_noise for smooth curriculum ramps.
        """
        cfg = self.cfg.experiment
        init_cfg = cfg.initialization
        grid_shape = tuple(cfg.grid_shape)
        num_nodes = int(cfg.num_nodes)
        init_line_distance = init_cfg.init_line_distance
        h, w = grid_shape
        mid = num_nodes // 2

        num_harmonics = int(init_cfg.get("fourier_num_harmonics", 4))
        raw_amps = init_cfg.get("fourier_amplitudes", [2.8, 0.7, 0.5, 0.2])
        amplitudes = jnp.array(list(raw_amps)[:num_harmonics], dtype=jnp.float32)

        # Explicit frequencies: e.g. [10, 11, 2, 3] — high-freq for tight rows, low-freq for global shape
        default_freqs = [10, 11, 2, 3]
        raw_freqs = init_cfg.get("fourier_frequencies", default_freqs)
        k_indices = jnp.array(list(raw_freqs)[:num_harmonics], dtype=jnp.float32)

        # Base phases fix the structural pattern (relative harmonic relationships).
        # In batch mode, add a single global offset so all samples share the same
        # serpentine structure but differ only in global orientation.
        raw_phases = list(init_cfg.get("fourier_phases", []))
        raw_phases = raw_phases + [0.0] * (num_harmonics - len(raw_phases))
        base_phases = jnp.array(raw_phases[:num_harmonics], dtype=jnp.float32)
        # noise can be a JAX dynamic value (float32 scalar) for smooth ramps
        phase_noise = noise if noise is not None else float(init_cfg.get("fourier_phase_noise", 1.0))
        noise_range = phase_noise * 2.0 * jnp.pi
        randomize_mode = init_cfg.get("fourier_randomize_phases", True)
        if not self.overfit_single_init or randomize_mode is True:
            global_offset = (jax.random.uniform(key, ()) - 0.5) * noise_range
            phases = base_phases + global_offset
        elif randomize_mode is False:
            phases = base_phases
        elif randomize_mode == "highest":
            # randomize only the highest-frequency harmonic
            idx = int(jnp.argmax(k_indices))
            offset = (jax.random.uniform(key, ()) - 0.5) * noise_range
            phases = base_phases.at[idx].add(offset)
        else:
            # integer index: randomize only that harmonic's phase
            idx = int(randomize_mode)
            offset = (jax.random.uniform(key, ()) - 0.5) * noise_range
            phases = base_phases.at[idx].add(offset)

        # Heading angle at each node: sum of sinusoids over normalized arc position
        t = jnp.linspace(0.0, 1.0, num_nodes)  # (N,)
        angle_components = amplitudes[None, :] * jnp.sin(
            2.0 * jnp.pi * k_indices[None, :] * t[:, None] + phases[None, :]
        )  # (N, H)
        angles = angle_components.sum(axis=1)  # (N,)

        steps = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1) * init_line_distance
        walk = jnp.concatenate([jnp.zeros((1, 2)), jnp.cumsum(steps[:-1], axis=0)], axis=0)

        seg = walk[mid + 1] - walk[mid]
        seg_angle = jnp.arctan2(seg[1], seg[0])
        cos_a, sin_a = jnp.cos(-seg_angle), jnp.sin(-seg_angle)
        rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
        walk = (walk - walk[mid]) @ rot.T
        # canonical winding: flip y if signed area negative so training always sees same orientation
        A = jnp.sum(walk[:-1, 0] * walk[1:, 1] - walk[1:, 0] * walk[:-1, 1])
        walk = jnp.where(A < 0, walk * jnp.array([1.0, -1.0]), walk)
        center_of_mass = walk.mean(axis=0)
        center = jnp.array([w / 2.0, h / 2.0])
        position = walk - center_of_mass + center

        nodes = self.nodes.__replace__(position=position)

        edges = self.edges
        if init_cfg.get("theta_pre_compute", False):
            src = edges.pairs[:, 0]
            tgt = edges.pairs[:, 1]
            deltas = displacement(grid_shape, nodes.position[tgt], nodes.position[src])
            edges = edges.__replace__(theta=jnp.arctan2(deltas[:, 1], deltas[:, 0]))

        return nodes, edges

    def _load_mnist_target(self) -> jnp.ndarray:
        import numpy as np
        from torchvision import datasets, transforms
        import jax.scipy.ndimage as jnd

        mnist_train = datasets.MNIST(
            root=str(Path(get_original_cwd()) / "data"),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        for image, label in mnist_train:
            if int(label) == self.target_digit:
                mnist_image = image.squeeze().numpy()
                break

        target = jnd.map_coordinates(
            mnist_image,
            jnp.meshgrid(
                jnp.linspace(0, 27, self.target_size),
                jnp.linspace(0, 27, self.target_size),
                indexing="ij",
            ),
            order=1,
        )
        return jnp.array(np.asarray(target), dtype=jnp.float32)

    def _load_target_pointcloud(self, num_points: int = 400) -> jnp.ndarray:
        import numpy as np

        image = np.array(self.target_image)  # (target_size, target_size)
        binary = image > 0.1

        use_skeleton = bool(self.cfg.experiment.loss.get("pointcloud_use_skeleton", True))
        erosion_iters = int(self.cfg.experiment.loss.get("pointcloud_skeleton_erosion_iters", 0))
        if use_skeleton:
            try:
                from skimage.morphology import skeletonize, erosion, disk
                thinned = binary
                if erosion_iters > 0:
                    for _ in range(erosion_iters):
                        thinned = erosion(thinned, disk(1))
                    if not thinned.any():
                        thinned = binary  # fallback if over-eroded
                mask = skeletonize(thinned)
            except ImportError:
                from scipy.ndimage import distance_transform_edt, binary_erosion
                thinned = binary
                for _ in range(erosion_iters):
                    thinned = binary_erosion(thinned)
                if not thinned.any():
                    thinned = binary
                dist = distance_transform_edt(thinned)
                mask = thinned & (dist >= 0.35 * dist.max())
        else:
            mask = binary

        rows, cols = np.where(mask)
        if len(rows) == 0:
            rows, cols = np.where(binary)
        if len(rows) == 0:
            rows, cols = np.where(np.ones_like(image, dtype=bool))

        rng = np.random.RandomState(42)
        if len(rows) >= num_points:
            idx = rng.choice(len(rows), num_points, replace=False)
        else:
            idx = rng.choice(len(rows), num_points, replace=True)

        h, w = tuple(self.cfg.experiment.grid_shape)
        scale_x = w / self.target_size
        scale_y = h / self.target_size
        x = cols[idx] * scale_x + scale_x / 2.0
        y = rows[idx] * scale_y + scale_y / 2.0
        return jnp.array(np.stack([x, y], axis=-1), dtype=jnp.float32)

    def _load_font_target(self) -> jnp.ndarray:
        import numpy as np
        from PIL import Image, ImageDraw
        import jax.scipy.ndimage as jnd

        render_size = 256
        char = self.target_character[:1] or "A"
        font = _load_pil_font(self.target_font, render_size)

        img = Image.new("L", (render_size, render_size), 0)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (render_size - w) / 2 - bbox[0]
        y = (render_size - h) / 2 - bbox[1]
        draw.text((x, y), char, font=font, fill=255)

        img_array = np.array(img, dtype=np.float32) / 255.0
        target = jnd.map_coordinates(
            img_array,
            jnp.meshgrid(
                jnp.linspace(0, render_size - 1, self.target_size),
                jnp.linspace(0, render_size - 1, self.target_size),
                indexing="ij",
            ),
            order=1,
        )
        return jnp.array(np.asarray(target), dtype=jnp.float32)

    def _load_image_target(self) -> jnp.ndarray:
        import numpy as np
        from PIL import Image
        import jax.scipy.ndimage as jnd

        img = Image.open(str(self.target_image_path)).convert("L")
        img_array = np.array(img, dtype=np.float32) / 255.0
        h_src, w_src = img_array.shape
        target = jnd.map_coordinates(
            img_array,
            jnp.meshgrid(
                jnp.linspace(0, h_src - 1, self.target_size),
                jnp.linspace(0, w_src - 1, self.target_size),
                indexing="ij",
            ),
            order=1,
        )
        return jnp.array(np.asarray(target), dtype=jnp.float32)

    def _save_pointcloud_png(self, points: jnp.ndarray, filename):
        import numpy as np
        h, w = tuple(self.cfg.experiment.grid_shape)
        pts = np.array(points)
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.scatter(pts[:, 0], pts[:, 1], s=2, c="white")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()

    def _transform_params(self, params: dict) -> dict:
        bending_rest_angles = jnp.tanh(params["bending_rest_angles_raw"]) * self.curvature_scale
        t = jax.nn.sigmoid(params["rest_lengths_raw"])
        rest_lengths = self.distance_min + t * (self.distance_max - self.distance_min)
        return {
            "bending_rest_angles": bending_rest_angles,
            "rest_lengths": rest_lengths,
        }

    def _simulate_timeseries(self, nodes, edges, extra_steps: int = 0,
                             feedback_mlp_params=None, target_bending_angles=None):
        total_steps = int(self.cfg.simulation.num_steps) + int(self.cfg.simulation.stability_steps) + extra_steps
        dt = self.cfg.simulation.dt
        scheme = self.scheme
        fields = self.fields
        pin_idx = self.pin_indices
        pin_positions = nodes.position[pin_idx]  # shape (2, 2), fixed per sample
        pin_edge_idx = self.pin_edge_idx
        use_feedback = feedback_mlp_params is not None and target_bending_angles is not None
        correction_scale = self.feedback_correction_scale if self.use_feedback else 1.0

        def scan_step(carry, _):
            nodes, edges, fields = carry
            if use_feedback:
                corrections = _compute_feedback_corrections(
                    feedback_mlp_params, edges, target_bending_angles, correction_scale
                )
                edges = replace(edges, bending_rest_angles=target_bending_angles + corrections)
            nodes, edges, fields = sim_step(nodes, edges, fields, dt, scheme)
            if self.pin_middle_nodes:
                new_pos = nodes.position.at[pin_idx].set(pin_positions)
                new_vel = nodes.velocity.at[pin_idx].set(jnp.zeros_like(pin_positions))
                nodes = replace(nodes, position=new_pos, velocity=new_vel)
                edges = replace(edges, theta=edges.theta.at[pin_edge_idx].set(0.0))
            fH, fW = fields.fluid_velocity.shape[1], fields.fluid_velocity.shape[2]
            H, W = fields.grid_shape
            fluid_vel_render = (
                jax.image.resize(fields.fluid_velocity, (2, H, W), method="linear")
                if (fH, fW) != (H, W) else fields.fluid_velocity
            )
            fields_for_render = replace(fields, f_grid=None, fluid_velocity=fluid_vel_render)
            return (nodes, edges, fields), (nodes, fields_for_render)

        (_, _, _), (nodes_ts, fields_ts) = jax.lax.scan(
            scan_step, (nodes, edges, fields), length=total_steps
        )
        return nodes_ts, fields_ts

    def _compute_eval_indices(self) -> list:
        n = int(self.cfg.simulation.num_steps)
        num_ef = self.num_eval_frames
        if num_ef == 1:
            return [n - 1]
        return [int(round(n // 2 - 1 + (n - 1 - (n // 2 - 1)) * i / (num_ef - 1))) for i in range(num_ef)]

    def _compute_stab_indices(self) -> list:
        n = int(self.cfg.simulation.num_steps)
        stab_steps = int(self.cfg.simulation.stability_steps)
        num_stab_frames = self.num_stability_frames
        if stab_steps <= 0 or num_stab_frames <= 1:
            return [n + stab_steps - 1] if stab_steps > 0 else []
        stab_end = n + stab_steps - 1
        return [int(round(n + (stab_end - n) * i / (num_stab_frames - 1))) for i in range(num_stab_frames)]

    def loss_fn(self, params, key, noise=None):
        effective_params = self._transform_params(params)
        feedback_mlp_params = params.get("feedback_mlp", None) if self.use_feedback else None
        target_bending_angles = effective_params["bending_rest_angles"] if self.use_feedback else None
        eval_indices = self._compute_eval_indices()
        stab_indices = self._compute_stab_indices()
        grid_shape = tuple(self.cfg.experiment.grid_shape)
        stab_weight = float(self.cfg.optimization.stability_weight)
        smoothness_weight = self.smoothness_weight
        max_dist_weight = self.max_dist_weight

        def single_loss(k):
            init_nodes, init_edges = self._make_init_state(k, noise)
            edges_updated = replace(init_edges, **effective_params)
            nodes_timeseries, _ = self._simulate_timeseries(
                init_nodes, edges_updated,
                feedback_mlp_params=feedback_mlp_params,
                target_bending_angles=target_bending_angles,
            )

            use_pca = self.use_pca_rotation

            def _centered(p, t):
                return p - jnp.mean(p, axis=0), t - jnp.mean(t, axis=0)

            if self.loss_mode == "pointcloud":
                def _pc_frame(t):
                    particles = nodes_timeseries.position[t]
                    if use_pca:
                        return _chamfer_loss_pca_aligned(particles, self.target_pointcloud)
                    pc, tc = _centered(particles, self.target_pointcloud)
                    return _chamfer_loss(pc, tc)
                shape_loss = jnp.mean(jnp.array([_pc_frame(t) for t in eval_indices]))
            elif self.loss_mode == "dcd":
                def _dcd_frame(t):
                    particles = nodes_timeseries.position[t]
                    if use_pca:
                        return _dcd_loss_pca_aligned(particles, self.target_pointcloud)
                    pc, tc = _centered(particles, self.target_pointcloud)
                    return _dcd_loss(pc, tc)
                shape_loss = jnp.mean(jnp.array([_dcd_frame(t) for t in eval_indices]))
            elif self.loss_mode == "hybrid":
                h, w = grid_shape
                grid_center = jnp.array([w / 2.0, h / 2.0])
                t_centered_pc = self.target_pointcloud - jnp.mean(self.target_pointcloud, axis=0)

                def _hybrid_frame_loss(t):
                    particles = nodes_timeseries.position[t]
                    nodes_t = jax.tree.map(lambda x: x[t], nodes_timeseries)

                    def _orient_loss(a):
                        rotated = _rotate_particles(particles, a)
                        pc_loss = _chamfer_loss(rotated, t_centered_pc)
                        rotated_nodes = nodes_t.__replace__(position=rotated + grid_center)
                        sdf_loss = jnp.mean((self.downsample_particles_to_grid(rotated_nodes) - self.target_image) ** 2)
                        return 0.5 * pc_loss + 0.5 * sdf_loss

                    if use_pca:
                        angle = _pca_rotation_angle(particles, self.target_pointcloud)
                        return jnp.minimum(_orient_loss(angle), _orient_loss(angle + jnp.pi))
                    return _orient_loss(jnp.array(0.0))

                shape_loss = jnp.mean(jnp.array([_hybrid_frame_loss(t) for t in eval_indices]))
            else:
                shape_loss = jnp.mean(jnp.array([
                    jnp.mean((self.downsample_particles_to_grid(
                        jax.tree.map(lambda x: x[t], nodes_timeseries)
                    ) - self.target_image) ** 2)
                    for t in eval_indices
                ]))

            pos_main = nodes_timeseries.position[eval_indices[-1]]
            # anchor 30 frames later: prevents bounce-back exploit where network learns
            # to oscillate and return to pos_main at each measurement point
            n_cfg = int(self.cfg.simulation.num_steps)
            stab_steps = int(self.cfg.simulation.stability_steps)
            total_steps = n_cfg + stab_steps
            anchor_t = min(eval_indices[-1] + self.stability_anchor_offset, total_steps - 1)
            use_anchor = anchor_t > eval_indices[-1]
            pos_anchor = nodes_timeseries.position[anchor_t]

            def _frame_stab(t):
                d = jnp.mean(displacement(grid_shape, nodes_timeseries.position[t], pos_main) ** 2)
                if use_anchor:
                    d = d + jnp.mean(displacement(grid_shape, nodes_timeseries.position[t], pos_anchor) ** 2)
                return d

            if stab_indices:
                stability_loss = jnp.mean(jnp.array([_frame_stab(t) for t in stab_indices]))
            else:
                stability_loss = _frame_stab(total_steps - 1)

            angles = effective_params["bending_rest_angles"]
            smoothness_loss = jnp.mean((angles[1:] - angles[:-1]) ** 2) if smoothness_weight > 0.0 else 0.0

            max_dist_loss = 0.0
            if max_dist_weight > 0.0 and self.loss_mode in ("dcd", "pointcloud", "hybrid"):
                def _sm_frame(t):
                    particles = nodes_timeseries.position[t]
                    if use_pca:
                        return _smooth_max_nn_pca_aligned(
                            particles, self.target_pointcloud, self.smooth_max_alpha,
                        )
                    pc, tc = _centered(particles, self.target_pointcloud)
                    return _smooth_max_nn(pc, tc, self.smooth_max_alpha)
                max_dist_loss = jnp.mean(jnp.array([_sm_frame(t) for t in eval_indices]))

            total_loss = (
                shape_loss
                + stab_weight * stability_loss
                + smoothness_weight * smoothness_loss
                + max_dist_weight * max_dist_loss
            )
            return total_loss, shape_loss, stability_loss

        if self.overfit_single_init:
            keys = jnp.stack([self.overfit_key])
        else:
            keys = jax.random.split(key, self.batch_size)
        all_total, all_shape, all_stab = jax.vmap(single_loss)(keys)
        return all_total.mean(), (all_shape.mean(), all_stab.mean())

    def _build_scheme(self):
        phys = self.cfg.get("physics", {})
        scheme_name = str(phys.get("scheme", "stable_cosserat"))
        base = _SCHEMES.get(scheme_name, COSSERAT_SCHEME_NO_FLUID)
        sp = phys.get(scheme_name, {})
        return base.__replace__(
            damping=float(sp.get("damping", base.damping)),
            stiffness_shear=float(sp.get("stiffness_shear", base.stiffness_shear)),
            stiffness_stretch=float(sp.get("stiffness_stretch", base.stiffness_stretch)),
            cycles_per_step=int(sp.get("cycles_per_step", base.cycles_per_step)),
            dt=float(self.cfg.simulation.dt),
            enable_steric=bool(phys.get("apply_steric_forces", base.enable_steric)),
            steric_strength=float(phys.get("steric_strength", base.steric_strength)),
            steric_sigma=float(phys.get("steric_sigma", base.steric_sigma)),
            steric_neighbor_skip=int(phys.get("steric_neighbor_skip", base.steric_neighbor_skip)),
            steric_scatter_value=float(phys.get("steric_scatter_value", base.steric_scatter_value)),
        )

    def _apply_curriculum_params(self, params_dict: dict):
        from omegaconf import OmegaConf
        from omegaconf import errors as oc_errors
        for dotkey, value in params_dict.items():
            try:
                OmegaConf.update(self.cfg, dotkey, value, merge=True)
            except (oc_errors.ConfigKeyError, KeyError):
                OmegaConf.update(self.cfg, f"experiment.{dotkey}", value, merge=True)
        self.scheme = self._build_scheme()

    def run(self):
        if self._multi_char:
            for char, exp in self.char_exps.items():
                print(f"\n{'='*60}")
                print(f"  Training character '{char}'")
                print(f"{'='*60}")
                exp.run()
                for step, loss in enumerate(exp.shape_loss_history):
                    self.tb.add_scalar(f"loss/shape_{char}", loss, global_step=step)

            per_char_avg = {}
            for char, exp in self.char_exps.items():
                last20 = exp.shape_loss_history[-20:]
                avg = sum(last20) / len(last20) if last20 else 9999.0
                per_char_avg[char] = avg
                print(f"[{char}] avg loss (last 20): {avg:.4f}")

            combined = sum(per_char_avg.values()) / len(per_char_avg)
            print(f"\nCombined avg loss (last 20): {combined:.4f}")
            self.tb.add_scalar("loss/shape_combined", combined, global_step=0)
            self.tb.close()
            self.save_results()
            return

        self._save_lr_schedule_plot()

        # Noise ramp: linearly increase noise from noise_ramp_start to fourier_phase_noise
        # over noise_ramp_steps. Passed as dynamic JAX float so no JIT recompile.
        _noise_end = float(self.cfg.experiment.initialization.fourier_phase_noise)
        _noise_start = float(self.cfg.experiment.initialization.get("noise_ramp_start", _noise_end))
        _noise_ramp_steps = int(self.cfg.optimization.get("noise_ramp_steps", 0))

        def _make_jit():
            return jax.jit(lambda params, key, noise: jax.value_and_grad(
                lambda p: self.loss_fn(p, key, noise), has_aux=True
            )(params))

        compute_loss_and_grad = None
        current_stage_idx = -1
        key = self.key
        pbar = tqdm(range(self.cfg.optimization.num_steps), desc="Optimizing filament")
        for step in pbar:
            # Advance curriculum: find highest stage whose start <= step
            active = -1
            for i, (start, _) in enumerate(self.curriculum_stages):
                if step >= start:
                    active = i
            if active != current_stage_idx or compute_loss_and_grad is None:
                current_stage_idx = active
                if active >= 0:
                    _, stage_params = self.curriculum_stages[active]
                    self._apply_curriculum_params(stage_params)
                    self.curriculum_events.append((step, stage_params))
                    pbar.write(f"[curriculum] stage {active} at step {step}: {stage_params}")
                compute_loss_and_grad = _make_jit()

            # Compute current noise (ramp if configured, else use curriculum/config value)
            if _noise_ramp_steps > 0:
                frac = min(1.0, float(step) / float(_noise_ramp_steps))
                current_noise = _noise_start + (_noise_end - _noise_start) * frac
            else:
                current_noise = float(self.cfg.experiment.initialization.fourier_phase_noise)
            noise_arg = jnp.array(current_noise, dtype=jnp.float32)

            key, subkey = jax.random.split(key)
            (loss, (shape_loss, stability_loss)), grads = compute_loss_and_grad(self.params, subkey, noise_arg)
            if jnp.isnan(loss):
                pbar.write(f"[NaN] loss is NaN at step {step}, stopping.")
                break
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)
            self.loss_history.append(float(loss))
            self.shape_loss_history.append(float(shape_loss))
            pbar.set_postfix({"loss": f"{loss:.4f}", "shape": f"{shape_loss:.4f}"})

            current_lr = float(self._lr_schedule(step))
            self.tb.add_scalar("loss/shape", float(shape_loss), global_step=step)
            self.tb.add_scalar("loss/stability", float(stability_loss), global_step=step)
            self.tb.add_scalar("lr", current_lr, global_step=step)
            self.tb.add_scalar("noise_ramp", float(current_noise), global_step=step)

            if self.viz_every > 0 and (step + 1) % self.viz_every == 0:
                self._save_progress(step + 1)
                progress_path = self.output_dir / f"progress_{step + 1:05d}.png"
                if progress_path.exists():
                    self.tb.add_image("progress", _load_png_chw(progress_path), global_step=step + 1)

        self.save_results()

    def save_results(self):
        if self._multi_char:
            for exp in self.char_exps.values():
                exp.save_results()
            return

        self._save_loss_plot(self.output_dir / "loss.png")

        effective_params = self._transform_params(self.params)
        fb_mlp = self.params.get("feedback_mlp", None) if self.use_feedback else None
        fb_target = effective_params["bending_rest_angles"] if self.use_feedback else None
        if self.overfit_single_init:
            eval_keys = [self.overfit_key]
        else:
            eval_keys = [jax.random.PRNGKey(i) for i in range(4)]
        density_paths = []
        eval_nodes_export = None
        eval_edges_export = None
        for i, k in enumerate(eval_keys):
            try:
                eval_nodes, eval_edges_base = self._make_init_state(k)
                final_edges = replace(eval_edges_base, **effective_params)
                nodes_timeseries, fields_timeseries = self._simulate_timeseries(
                    eval_nodes, final_edges,
                    feedback_mlp_params=fb_mlp, target_bending_angles=fb_target,
                )
                final_nodes = jax.tree.map(lambda x: x[-1], nodes_timeseries)
                density = self.downsample_particles_to_grid(final_nodes)
                path = self.output_dir / f"final_density_{i}.png"
                self._save_density_png(density, path)
                density_paths.append(path)
                if i == 0:
                    self._save_comparison_png(density, self.output_dir / "comparison.png")
                    self._save_steric_field_png(fields_timeseries, final_nodes, self.output_dir / "steric_field.png")
                    try:
                        self._save_keyframe_grid(nodes_timeseries, self.output_dir / "keyframes.png")
                    except Exception as e:
                        print(f"[warning] keyframe grid failed: {e}")
                    eval_nodes_export, eval_edges_export = eval_nodes, eval_edges_base
                # Per-seed fluid simulation video (all seeds, not just i==0)
                try:
                    extra_viz = int(self.cfg.simulation.get("extra_viz_steps", 80))
                    viz_nodes_ts, viz_fields_ts = self._simulate_timeseries(
                        eval_nodes, final_edges, extra_steps=extra_viz,
                        feedback_mlp_params=fb_mlp, target_bending_angles=fb_target,
                    )
                    self._save_final_gif(viz_nodes_ts, viz_fields_ts,
                                         filename=str(self.output_dir / f"final_simulation_fluid_{i}.mp4"))
                except Exception as e:
                    print(f"[warning] fluid video seed {i} failed (NaN?): {e}")
            except Exception as e:
                print(f"[warning] eval key {i} failed (NaN?): {e}")

        try:
            self._export_graph_json(effective_params, eval_nodes_export, eval_edges_export)
        except Exception as e:
            print(f"[warning] graph export failed: {e}")
        for p in density_paths:
            print(f"Result: {p}")

        final_step = len(self.loss_history)
        comparison_path = self.output_dir / "comparison.png"
        if comparison_path.exists():
            self.tb.add_image("final/comparison", _load_png_chw(comparison_path), global_step=final_step)
        for i, p in enumerate(density_paths):
            if p.exists():
                self.tb.add_image(f"final/density_seed{i}", _load_png_chw(p), global_step=final_step)
        loss_path = self.output_dir / "loss.png"
        if loss_path.exists():
            self.tb.add_image("final/loss_plot", _load_png_chw(loss_path), global_step=final_step)
        keyframes_path = self.output_dir / "keyframes.png"
        if keyframes_path.exists():
            self.tb.add_image("final/keyframes", _load_png_chw(keyframes_path), global_step=final_step)
        steric_path = self.output_dir / "steric_field.png"
        if steric_path.exists():
            self.tb.add_image("final/steric_field", _load_png_chw(steric_path), global_step=final_step)
        self.tb.close()
        print(f"[tensorboard] run: tensorboard --logdir {self.output_dir / 'tb'}")

    def _export_graph_json(self, effective_params: dict, eval_nodes=None, eval_edges_base=None):
        """Serialize the optimized line filament as WebGPU-compatible graph JSON.

        Maps per-edge (rest_lengths, bending_rest_angles) into per-node
        (local_line_distance, local_curvature.next). For line topology: edge k
        connects node k -> k+1; bending pair k sits at node k+1 (k in 0..N-3).
        """
        src_nodes = eval_nodes if eval_nodes is not None else self.nodes
        src_edges = eval_edges_base if eval_edges_base is not None else self.edges
        num_nodes = int(self.cfg.experiment.num_nodes)
        bending = [float(x) for x in effective_params["bending_rest_angles"]]
        rest_len = [float(x) for x in effective_params["rest_lengths"]]
        positions = [[float(p[0]), float(p[1])] for p in src_nodes.position]
        grid_shape = [int(d) for d in self.cfg.experiment.grid_shape]

        init_cfg = self.cfg.experiment.initialization
        init_mode = str(init_cfg.get("init_mode", "random_walk"))

        metadata = {
            "digit": int(self.target_digit),
            "font": self.target_font,
            "character": self.target_character if self.target_font else None,
            "num_nodes": num_nodes,
            "init_line_distance": float(init_cfg.init_line_distance),
            "dt": float(self.cfg.simulation.dt),
            "num_steps": int(self.cfg.simulation.num_steps),
            "substeps": int(self.scheme.cycles_per_step),
            "enable_fluid": bool(self.scheme.enable_fluid),
            "stiffness_stretch": float(self.scheme.stiffness_stretch),
            "stiffness_shear": float(self.scheme.stiffness_shear),
            "damping": float(self.scheme.damping),
            "viscosity": float(self.scheme.viscosity),
            "max_fluid_velocity": float(self.scheme.max_fluid_velocity),
            "lambda_trt": float(self.scheme.lambda_trt),
            "ibm_iterations": int(self.scheme.ibm_iterations),
            "ibm_relaxation": float(self.scheme.ibm_relaxation),
            "node_mass": float(self.scheme.node_mass),
        }

        if self.use_feedback:
            fb_mlp = self.params.get("feedback_mlp", None)
            if fb_mlp is not None:
                fb_cfg_e = self.cfg.experiment.get("feedback_control", {})
                metadata["feedback_mlp"] = {
                    "enabled": True,
                    "hidden_dim": int(fb_cfg_e.get("hidden_dim", 8)),
                    "correction_scale": self.feedback_correction_scale,
                    "W1": [list(row) for row in fb_mlp["W1"].tolist()],
                    "b1": list(fb_mlp["b1"].tolist()),
                    "W2": list(fb_mlp["W2"].tolist()),
                    "b2": float(fb_mlp["b2"]),
                }

        if init_mode == "fourier_series":
            num_harmonics = int(init_cfg.get("fourier_num_harmonics", 4))
            raw_amps = list(init_cfg.get("fourier_amplitudes", [2.8, 0.7, 0.5, 0.2]))[:num_harmonics]
            raw_freqs = list(init_cfg.get("fourier_frequencies", [10, 11, 2, 3]))[:num_harmonics]
            raw_phases = list(init_cfg.get("fourier_phases", []))
            raw_phases = (raw_phases + [0.0] * num_harmonics)[:num_harmonics]
            metadata.update({
                "fourier_num_harmonics": num_harmonics,
                "fourier_frequencies": [float(f) for f in raw_freqs],
                "fourier_amplitudes": [float(a) for a in raw_amps],
                "fourier_phases": [float(p) for p in raw_phases],
                "canonical_init": bool(init_cfg.get("canonical_init", False)),
            })

        # Per-node theta: use the actual JAX edge thetas (set by init_nodes("line"),
        # never recomputed after the random-walk position overwrite).
        # This matches the value that enters the first PBD step in JAX exactly.
        edge_thetas = [float(t) for t in src_edges.theta]  # length N-1
        node_thetas = edge_thetas + [edge_thetas[-1] if edge_thetas else 0.0]

        nodes_json = []
        for i in range(num_nodes):
            prev_id = i - 1 if i > 0 else None
            next_id = i + 1 if i < num_nodes - 1 else None

            if next_id is not None:
                line_dist = rest_len[i]
                curv_next = bending[i] if i < num_nodes - 2 else 0.0
            else:
                line_dist = 0.0
                curv_next = None

            prev_curv = 0.0 if prev_id is not None else None

            nodes_json.append({
                "id": i,
                "position": positions[i],
                "theta": node_thetas[i],
                "local_line_distance": line_dist,
                "next": next_id,
                "prev": prev_id,
                "left": None,
                "right": None,
                "local_curvature": {
                    "next": curv_next,
                    "prev": prev_curv,
                    "left": None,
                    "right": None,
                },
            })

        graph = {"grid_shape": grid_shape, "metadata": metadata, "nodes": nodes_json}

        if self.target_image_path:
            char_name = self.target_image_path.stem
        elif self.target_font and self.target_character.isdigit():
            char_name = f"digit_{self.target_character}"
        elif self.target_font:
            char_name = self.target_character.lower()
        else:
            char_name = f"digit_{self.target_digit}"
        out_path = self.output_dir / f"{char_name}.json"
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)
        print(f"[export] {out_path}")

        if bool(self.cfg.experiment.get("export_to_digits", True)):
            webgpu_root = Path(__file__).resolve().parents[3]
            digits_dir = webgpu_root / "DIGITS"
            digits_dir.mkdir(exist_ok=True)
            digits_path = digits_dir / f"{char_name}.json"
            with open(digits_path, "w") as f:
                json.dump(graph, f, indent=2)
            print(f"[export] {digits_path}")

        # Drop a copy at webgpu repo root as the default graph.json load target.
        # default_path = webgpu_root / "graph.json"
        # with open(default_path, "w") as f:
        #     json.dump(graph, f, indent=2)

    def _save_progress(self, step: int):
        effective_params = self._transform_params(self.params)
        fb_mlp = self.params.get("feedback_mlp", None) if self.use_feedback else None
        fb_target = effective_params["bending_rest_angles"] if self.use_feedback else None
        if self.overfit_single_init:
            viz_keys = [self.overfit_key]
        else:
            viz_keys = [jax.random.PRNGKey(s) for s in [0, 1, 2]]
        densities = []
        for k in viz_keys:
            init_nodes, init_edges = self._make_init_state(k)
            edges_updated = replace(init_edges, **effective_params)
            nodes_ts, _ = self._simulate_timeseries(
                init_nodes, edges_updated,
                feedback_mlp_params=fb_mlp, target_bending_angles=fb_target,
            )
            densities.append(self.downsample_particles_to_grid(jax.tree.map(lambda x: x[-1], nodes_ts)))

        ncols = 1 + len(densities)
        fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.5))
        if ncols == 1:
            axes = [axes]
        axes[0].imshow(self.target_image, cmap="gray", origin="upper", interpolation="nearest")
        axes[0].set_title("Target")
        axes[0].axis("off")
        viz_seeds = [0, 1, 2] if not self.overfit_single_init else [int(self.overfit_key[1])]
        for i, (d, k) in enumerate(zip(densities, viz_keys)):
            label = f"seed {viz_seeds[i]}" if not self.overfit_single_init else f"overfit seed {viz_seeds[0]}"
            axes[i + 1].imshow(d, cmap="gray", origin="upper", interpolation="nearest")
            axes[i + 1].set_title(label)
            axes[i + 1].axis("off")
        plt.suptitle(f"step {step}  shape_loss={self.shape_loss_history[-1]:.4f}", fontsize=10)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"progress_{step:05d}.png", dpi=100, bbox_inches="tight")
        plt.close()

        self._save_loss_plot(self.output_dir / "loss.png")

    def _save_loss_plot(self, filename):
        fig, ax = plt.subplots()
        ax.plot(self.shape_loss_history, label="Shape Loss")
        for step, params_dict in self.curriculum_events:
            ax.axvline(x=step, color="red", linestyle="--", alpha=0.7, linewidth=1.0)
            label = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in params_dict.items())
            ax.text(step + 0.5, 0.97, label, transform=ax.get_xaxis_transform(),
                    fontsize=7, color="red", va="top", rotation=90)
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Loss")
        ax.set_title("Filament Folding")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_density_png(self, density, filename):
        plt.figure(figsize=(3.2, 3.2))
        plt.imshow(density, cmap="gray", origin="upper", interpolation="nearest")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()

    def _save_comparison_png(self, density, filename):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(self.target_image, cmap="gray", origin="upper", interpolation="nearest")
        axes[0].set_title("Target Image")
        axes[0].axis("off")
        axes[1].imshow(density, cmap="gray", origin="upper", interpolation="nearest")
        axes[1].set_title("Final Result")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

    def _save_steric_field_png(self, fields_timeseries, final_nodes, filename):
        import numpy as np
        steric = np.array(fields_timeseries.steric[-1])  # (H, W) at last sim step
        positions = np.array(final_nodes.position)       # (N, 2)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        im0 = axes[0].imshow(steric, cmap="hot", origin="upper")
        axes[0].scatter(positions[:, 0], positions[:, 1], s=1, c="cyan", alpha=0.6)
        axes[0].set_title(f"Steric field  (sigma={self.scheme.steric_sigma})")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        steric_log = np.log1p(steric)
        im1 = axes[1].imshow(steric_log, cmap="hot", origin="upper")
        axes[1].scatter(positions[:, 0], positions[:, 1], s=1, c="cyan", alpha=0.6)
        axes[1].set_title("Steric field (log1p)")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[steric] {filename}")

    def _save_keyframe_grid(self, nodes_timeseries, filename):
        """Save density images at shape eval frames and stability frames side by side."""
        eval_indices = self._compute_eval_indices()
        stab_indices = self._compute_stab_indices()
        frames = (
            [(f"shape t={t}", t) for t in eval_indices]
            + [(f"stab t={t}", t) for t in stab_indices]
        )
        if not frames:
            return
        ncols = 1 + len(frames)  # target + keyframes
        fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.2))
        axes[0].imshow(self.target_image, cmap="gray", origin="upper", interpolation="nearest")
        axes[0].set_title("target", fontsize=8)
        axes[0].axis("off")
        for ax, (label, t) in zip(axes[1:], frames):
            nodes_t = jax.tree.map(lambda x: x[t], nodes_timeseries)
            density = self.downsample_particles_to_grid(nodes_t)
            ax.imshow(density, cmap="gray", origin="upper", interpolation="nearest")
            ax.set_title(label, fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[keyframes] {filename}")

    def _save_final_gif(self, nodes_timeseries, fields_timeseries, filename: str | None = None):
        out = filename if filename is not None else str(self.output_dir / "final_simulation.mp4")
        animate(
            nodes_timeseries,
            fields_timeseries,
            out,
            subsample=1,
            animate_fluid_velocity=True,
        )
