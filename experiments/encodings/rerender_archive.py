"""Re-render QD archive videos at a custom ``simulation.num_steps``.

Loads a saved QD run (its ``.hydra/config.yaml`` and ``archive.npz``),
overrides the render length, and re-invokes ``_render_diverse_animations``
into a subdirectory so the original mp4s are not overwritten.

Usage:
    cd experiments
    uv run python rerender_archive.py \
        --run-dir "outputs/qd/2026-04-04/20-55-40 - really nice swimmers" \
        --num-steps 1500

Optional flags:
    --subdir NAME            Output subdir inside run-dir (default: longer_<N>steps)
    --sample-grid ROWS COLS  Override render_sample_grid (e.g. 2 2 for a quick test)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Make ``experiments`` importable when run as a plain script from that dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.qd import QDExperiment


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Path to a saved QD output directory containing .hydra/ and archive.npz")
    ap.add_argument("--num-steps", type=int, default=1500,
                    help="simulation.num_steps to use for rendering (default: 1500)")
    ap.add_argument("--subdir", type=str, default=None,
                    help="Output subdir inside run-dir (default: longer_<N>steps)")
    ap.add_argument("--sample-grid", type=int, nargs=2, default=None,
                    metavar=("ROWS", "COLS"),
                    help="Override render_sample_grid (e.g. --sample-grid 2 2 for a quick test)")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run-dir not found: {run_dir}")

    cfg_path = run_dir / ".hydra" / "config.yaml"
    archive_path = run_dir / "archive.npz"
    if not cfg_path.is_file():
        raise SystemExit(f"missing config: {cfg_path}")
    if not archive_path.is_file():
        raise SystemExit(f"missing archive: {archive_path}")

    cfg = OmegaConf.load(cfg_path)
    old_steps = int(cfg.simulation.get("num_steps", 400))
    cfg.simulation.num_steps = int(args.num_steps)
    if args.sample_grid is not None:
        cfg.qd.render_sample_grid = list(args.sample_grid)

    subdir = args.subdir or f"longer_{args.num_steps}steps"
    out_dir = run_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading archive from {archive_path}")
    data = np.load(archive_path)
    archive_fitness = data["fitness"]
    archive_neurons = data["neurons"]
    archive_conns = data["conns"]

    sample_grid = list(cfg.qd.get("render_sample_grid", [10, 10]))
    print(
        f"Re-rendering archive {archive_fitness.shape} "
        f"({old_steps} -> {args.num_steps} steps, sample grid {sample_grid}) "
        f"into {out_dir}"
    )

    exp = QDExperiment(cfg, output_dir=out_dir)
    exp.setup()
    exp._render_diverse_animations(
        archive_fitness, archive_neurons, archive_conns, sample_grid
    )


if __name__ == "__main__":
    main()
