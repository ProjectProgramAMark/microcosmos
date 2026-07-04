"""Text composition experiment: load pre-trained per-character curvatures and
tile them as disconnected filament chains spelling a string.

Each character is one chain. The text string is split on whitespace into rows;
each row is laid out horizontally. Per-character JSON files (produced by the
filament_folding experiment) supply init positions, edge thetas, rest_lengths,
bending_rest_angles, and the feedback MLP. The chains are placed into a single
combined Nodes/Edges struct so steric forces can act between letters.
"""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from experiments.base_experiment import Experiment
from microcosmos.graph import compute_bending_pairs, make_edges
from microcosmos.simulate import step as sim_step
from microcosmos.solver.config import COSSERAT_SCHEME_NO_FLUID, PBD_SCHEME_NO_FLUID
from microcosmos.structs.edges import Edges
from microcosmos.structs.fields import Fields
from microcosmos.structs.nodes import Nodes

_SCHEMES = {
    "pbd": PBD_SCHEME_NO_FLUID,
    "stable_cosserat": COSSERAT_SCHEME_NO_FLUID,
}


def _load_char_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_chain_from_json(data: dict) -> dict:
    """Pull init positions, edge thetas, rest_lengths, bending_rest_angles, MLP params."""
    nodes_data = data["nodes"]
    N = len(nodes_data)
    positions = np.array([n["position"] for n in nodes_data], dtype=np.float32)  # (N, 2)
    # edge i goes node i -> i+1; node i.theta is the per-edge theta written at export
    thetas = np.array([nodes_data[i]["theta"] for i in range(N - 1)], dtype=np.float32)  # (N-1,)
    rest_lengths = np.array(
        [nodes_data[i]["local_line_distance"] for i in range(N - 1)], dtype=np.float32
    )  # (N-1,)
    # bending pair k sits at node k+1 with rest_angle = node[k].local_curvature.next, for k in 0..N-3
    bending_rest_angles = np.array(
        [nodes_data[i]["local_curvature"]["next"] for i in range(N - 2)], dtype=np.float32
    )  # (N-2,)

    src_grid = data.get("grid_shape", [256, 256])  # (h, w)
    src_center = np.array([src_grid[1] / 2.0, src_grid[0] / 2.0], dtype=np.float32)

    meta = data.get("metadata", {})
    fb = meta.get("feedback_mlp", {})
    feedback = None
    if fb.get("enabled", False):
        feedback = {
            "W1": np.array(fb["W1"], dtype=np.float32),
            "b1": np.array(fb["b1"], dtype=np.float32),
            "W2": np.array(fb["W2"], dtype=np.float32),
            "b2": np.float32(fb["b2"]),
            "correction_scale": float(fb.get("correction_scale", 0.4)),
        }

    return {
        "N": N,
        "positions": positions,
        "thetas": thetas,
        "rest_lengths": rest_lengths,
        "bending_rest_angles": bending_rest_angles,
        "src_center": src_center,
        "feedback": feedback,
    }


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if not p.is_absolute():
        p = Path(get_original_cwd()) / p
    return p


class TextFilamentsExperiment(Experiment):
    def setup(self):
        cfg = self.cfg.experiment
        text = str(cfg.text)
        # split on whitespace into rows ("ATAT ATAT" -> ["ATAT", "ATAT"])
        rows = [r for r in text.split() if r]
        if not rows:
            raise ValueError("text is empty")

        char_paths_raw = OmegaConf.to_container(cfg.char_paths, resolve=True) or {}
        char_paths = {k.upper(): v for k, v in char_paths_raw.items()}

        unique_chars = {ch.upper() for row in rows for ch in row}
        missing = sorted(unique_chars - set(char_paths))
        if missing:
            raise ValueError(
                f"text {text!r} needs chars {missing} not present in char_paths "
                f"(have: {sorted(char_paths)}). Train them via filament_folding or remove from text."
            )
        char_data: dict[str, dict] = {}
        for ch in unique_chars:
            char_data[ch] = _extract_chain_from_json(_load_char_json(_resolve_path(char_paths[ch])))

        # all chains must share node count
        Ns = {ch: d["N"] for ch, d in char_data.items()}
        if len(set(Ns.values())) != 1:
            raise ValueError(f"All character JSONs must have identical num_nodes; got {Ns}")
        per_letter_N = next(iter(Ns.values()))

        cfg_grid = tuple(int(x) for x in cfg.grid_shape)
        cell_w = float(cfg.cell_width)
        cell_h = float(cfg.cell_height)

        # center each row horizontally, stack rows vertically and center the block
        n_rows = len(rows)
        max_cols = max(len(r) for r in rows)

        # auto-grow grid_shape so the block fits with a margin; periodic-boundary wrap
        # otherwise truncates top/bottom rows when the user adds more words.
        margin = int(cfg.get("auto_grid_margin", 64))
        needed_h = int(n_rows * cell_h) + margin
        needed_w = int(max_cols * cell_w) + margin
        h = max(cfg_grid[0], needed_h)
        w = max(cfg_grid[1], needed_w)
        if (h, w) != cfg_grid:
            print(f"[text_filaments] auto-expanded grid_shape {cfg_grid} -> ({h}, {w}) "
                  f"to fit {n_rows} rows × {max_cols} cols")
        grid_shape = (h, w)
        block_h = n_rows * cell_h
        y_top = (h - block_h) / 2.0 + cell_h / 2.0

        slot_chars: list[str] = []
        slot_centers: list[tuple[float, float]] = []
        for row_idx, row in enumerate(rows):
            row_len = len(row)
            row_w = row_len * cell_w
            x_left = (w - row_w) / 2.0 + cell_w / 2.0
            cy = y_top + row_idx * cell_h
            for col_idx, ch in enumerate(row):
                cx = x_left + col_idx * cell_w
                slot_chars.append(ch.upper())
                slot_centers.append((cx, cy))

        K = len(slot_chars)
        N = per_letter_N
        total_nodes = K * N

        all_positions: list[np.ndarray] = []
        all_thetas: list[np.ndarray] = []
        all_rest_lengths: list[np.ndarray] = []
        all_bending: list[np.ndarray] = []
        all_fb: list[dict | None] = []
        for ch, (cx, cy) in zip(slot_chars, slot_centers):
            d = char_data[ch]
            offset = np.array([cx, cy], dtype=np.float32) - d["src_center"]
            all_positions.append(d["positions"] + offset)
            all_thetas.append(d["thetas"])
            all_rest_lengths.append(d["rest_lengths"])
            all_bending.append(d["bending_rest_angles"])
            all_fb.append(d["feedback"])

        edge_pairs, departure_angles = make_edges(total_nodes, graph="line", num_instances=K)
        bending_pairs, _ = compute_bending_pairs(edge_pairs, departure_angles, total_nodes)

        positions_arr = jnp.array(np.concatenate(all_positions, axis=0))
        velocity = jnp.zeros_like(positions_arr)
        nodes = Nodes(
            position=positions_arr,
            velocity=velocity,
            color_bend=jnp.arange(total_nodes) % 3,
            debug_vector=jnp.zeros_like(positions_arr),
        )
        thetas_arr = jnp.array(np.concatenate(all_thetas, axis=0))
        rest_lengths_arr = jnp.array(np.concatenate(all_rest_lengths, axis=0))
        bending_rest_arr = jnp.array(np.concatenate(all_bending, axis=0))
        bending_stiffness = jnp.full(
            bending_pairs.shape[0], float(cfg.get("bending_stiffness", 0.5)), dtype=jnp.float32
        )

        edges = Edges(
            pairs=edge_pairs,
            theta=thetas_arr,
            rest_lengths=rest_lengths_arr,
            bending_pairs=bending_pairs,
            bending_rest_angles=bending_rest_arr,
            bending_stiffness=bending_stiffness,
        )

        weights = jnp.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
        self.fields = Fields(
            grid_shape=grid_shape,
            steric=jnp.zeros(grid_shape),
            fluid_velocity=jnp.zeros((2, h, w)),
            f_grid=jnp.zeros((9, h, w)).at[:].set(weights[:, None, None]),
        )

        self.nodes = nodes
        self.edges = edges
        self.grid_shape = grid_shape
        self.scheme = self._build_scheme()
        self.slot_chars = slot_chars
        self.per_letter_N = N
        self.num_chains = K

        # stack per-chain feedback MLP params (one slot of params per letter slot)
        self.use_feedback = bool(cfg.get("use_feedback", True)) and all(fb is not None for fb in all_fb)
        if self.use_feedback:
            self.mlp_W1 = jnp.stack([jnp.asarray(fb["W1"]) for fb in all_fb])
            self.mlp_b1 = jnp.stack([jnp.asarray(fb["b1"]) for fb in all_fb])
            self.mlp_W2 = jnp.stack([jnp.asarray(fb["W2"]) for fb in all_fb])
            self.mlp_b2 = jnp.stack([jnp.asarray(fb["b2"]) for fb in all_fb])
            self.mlp_corr_scale = jnp.array(
                [fb["correction_scale"] for fb in all_fb], dtype=jnp.float32
            )
            self.target_bending = bending_rest_arr.reshape(K, N - 2)
            print(f"[text_filaments] feedback MLP enabled on {K} chains")
        else:
            print(f"[text_filaments] feedback MLP disabled (any chain missing it: "
                  f"{any(fb is None for fb in all_fb)})")

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

    def loss_fn(self, *args, **kwargs):
        return None

    def _simulate(self, num_steps: int, dt: float):
        scheme = self.scheme
        K = self.num_chains
        N = self.per_letter_N

        if self.use_feedback:
            W1 = self.mlp_W1
            b1 = self.mlp_b1
            W2 = self.mlp_W2
            b2 = self.mlp_b2
            corr_scale = self.mlp_corr_scale
            target_bending = self.target_bending
            target_flat = target_bending.reshape(-1)

            def per_chain_correction(W1c, b1c, W2c, b2c, scale_c, error_c):
                # error_c: (N-2,)
                error_left = jnp.concatenate([error_c[:1], error_c[:-1]])
                error_right = jnp.concatenate([error_c[1:], error_c[-1:]])
                inputs = jnp.stack([error_left, error_c, error_right], axis=-1)

                def fwd(x):
                    h = jnp.tanh(x @ W1c + b1c)
                    return jnp.tanh(jnp.dot(h, W2c) + b2c)

                return jax.vmap(fwd)(inputs) * scale_c

        def scan_step(carry, _):
            nodes, edges, fields = carry
            if self.use_feedback:
                e_in = edges.bending_pairs[:, 0]
                e_out = edges.bending_pairs[:, 1]
                current_bending = edges.theta[e_out] - edges.theta[e_in]
                error = (current_bending - target_flat + jnp.pi) % (2 * jnp.pi) - jnp.pi
                error_per_chain = error.reshape(K, N - 2)
                corrections = jax.vmap(per_chain_correction)(
                    W1, b1, W2, b2, corr_scale, error_per_chain
                ).reshape(-1)
                edges = replace(edges, bending_rest_angles=target_flat + corrections)
            nodes, edges, fields = sim_step(nodes, edges, fields, dt, scheme)

            fH, fW = fields.fluid_velocity.shape[1], fields.fluid_velocity.shape[2]
            H, W_ = fields.grid_shape
            fluid_vel_render = (
                jax.image.resize(fields.fluid_velocity, (2, H, W_), method="linear")
                if (fH, fW) != (H, W_)
                else fields.fluid_velocity
            )
            fields_for_render = replace(fields, f_grid=None, fluid_velocity=fluid_vel_render)
            return (nodes, edges, fields), (nodes, fields_for_render)

        (_, _, _), (nodes_ts, fields_ts) = jax.lax.scan(
            scan_step, (self.nodes, self.edges, self.fields), length=num_steps
        )
        return nodes_ts, fields_ts

    def run(self):
        dt = float(self.cfg.simulation.dt)
        num_steps = int(self.cfg.simulation.num_steps)
        nodes_ts, fields_ts = self._simulate(num_steps, dt)
        jax.block_until_ready(nodes_ts.position)

        sub = max(1, int(self.cfg.get("rendering", {}).get("subsample", 1)))
        world_size = int(self.cfg.get("rendering", {}).get("world_size", 1024))
        nodes_sub = jax.tree.map(lambda x: x[::sub], nodes_ts)
        fields_sub = jax.tree.map(lambda x: x[::sub], fields_ts)
        frames_uint8 = self._render_blue_frames(nodes_sub, fields_sub, world_size)

        # PNGs: init = first sim frame (steric is ~zero), final = last
        Image.fromarray(np.array(frames_uint8[0])).save(self.output_dir / "text_init.png")
        Image.fromarray(np.array(frames_uint8[-1])).save(self.output_dir / "text_final.png")

        self._write_mp4(frames_uint8, self.output_dir / "text_filament.mp4")
        print(f"[done] {self.output_dir}")

    @staticmethod
    def _render_blue_frames(nodes_sub, fields_sub, world_size: int):
        """Custom renderer: blue steric-gradient background + bright cyan particles. No debug overlay."""
        positions = nodes_sub.position  # (T, N, 2)
        steric = fields_sub.steric  # (T, H, W)
        T = positions.shape[0]
        H, W = steric.shape[1], steric.shape[2]

        # log-compress huge steric values, normalize by global max, gamma to brighten glow
        steric_log = jnp.log1p(jnp.clip(steric, 0.0, None))
        norm = jnp.max(steric_log) + 1e-8
        steric_norm = jnp.clip(steric_log / norm, 0.0, 1.0) ** 0.4

        bg_color = jnp.array([0.03, 0.06, 0.15])   # dark blue background
        mid_color = jnp.array([0.20, 0.55, 1.00])  # bright blue at steric peaks
        diff = mid_color - bg_color
        render_tex = bg_color[None, None, None, :] + steric_norm[..., None] * diff[None, None, None, :]

        pixel_pos = jnp.floor(positions).astype(jnp.int32)
        px = jnp.clip(pixel_pos[..., 0], 0, W - 1)
        py = jnp.clip(pixel_pos[..., 1], 0, H - 1)
        N = px.shape[1]
        frame_idx = jnp.broadcast_to(jnp.arange(T)[:, None], (T, N))
        particle_color = jnp.array([0.75, 0.90, 1.0])
        particle_colors = jnp.broadcast_to(particle_color[None, None, :], (T, N, 3))
        render_tex = render_tex.at[(frame_idx, py, px)].set(particle_colors)

        render_tex = jnp.clip(render_tex * 255.0, 0, 255).astype(jnp.uint8)
        out_h = world_size * H // max(H, W)
        out_w = world_size * W // max(H, W)
        return jax.image.resize(render_tex, (T, out_h, out_w, 3), method="nearest")

    @staticmethod
    def _write_mp4(frames_uint8, out_path: Path, fps: int = 30):
        frames_np = np.array(frames_uint8, dtype=np.uint8)
        H, W = frames_np.shape[1], frames_np.shape[2]
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{W}x{H}",
                "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                str(out_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        for frame in frames_np:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
