"""Build report-ready R15--R20 result figures from frozen artifacts.

The composite figure uses only the frozen comparison table.  The founder-level
figure joins the selected coherent repeat's ``private.pair_deltas`` to the
corresponding frozen manifest, preserving manifest order.  The stored metrics
do not contain founder-specific sham and injury deltas, so this script does not
attempt to reconstruct them.
"""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "ShinkaEvolve/examples/evo2_ecosystem/results"
FROZEN_SUMMARY = RESULTS / "evo2-r15-r20-final-artifacts-20260716"
COMPARISON_CSV = FROZEN_SUMMARY / "comparison.csv"

DEVELOPMENT_METRICS = (
    RESULTS
    / "evo2-exploratory-r20-gen16-r18-development-20260716"
    / "baselines/r15_gen18/metrics.json"
)
SEALED_METRICS = (
    RESULTS
    / "evo2-exploratory-r15-gen18-r18-sealed-20260716"
    / "evaluation_r15_gen18/metrics.json"
)
DEVELOPMENT_MANIFEST = (
    ROOT
    / "microcosmos/experiments/evo2_ecosystem/r18_artifacts/manifests"
    / "development_early_injury.json"
)
SEALED_MANIFEST = (
    ROOT
    / "microcosmos/experiments/evo2_ecosystem/r18_artifacts/manifests"
    / "sealed_early_injury.json"
)

OPERATORS = (
    "parametric_conservative",
    "parametric_standard",
    "parametric_exploratory",
    "structural",
    "mixed",
)
OPERATOR_LABELS = {
    "parametric_conservative": "conservative",
    "parametric_standard": "standard",
    "parametric_exploratory": "exploratory",
    "structural": "structural",
    "mixed": "mixed",
}
OPERATOR_COLORS = {
    "parametric_conservative": "#4C78A8",
    "parametric_standard": "#F58518",
    "parametric_exploratory": "#B279A2",
    "structural": "#54A24B",
    "mixed": "#E45756",
}

ROW_COLORS = {
    ("R20 gen16", "hard training"): "#6B7280",
    ("R20 gen16", "R18 development"): "#6B7280",
    ("fixed standard", "R18 development"): "#D97706",
    ("fixed conservative", "R18 development"): "#D97706",
    ("R15 gen18", "R18 development"): "#2563EB",
    ("matched sparse", "R18 development"): "#7C3AED",
    ("no-crisis ablation", "R18 development"): "#7C3AED",
    ("R15 gen18", "R18 sealed"): "#DC2626",
}

DISPLAY_LABELS = {
    ("R20 gen16", "hard training"): "R20 gen 16: hard training",
    ("R20 gen16", "R18 development"): "R20 gen 16: development",
    ("fixed standard", "R18 development"): "Fixed standard: development",
    ("fixed conservative", "R18 development"): "Fixed conservative: development",
    ("R15 gen18", "R18 development"): "R15 gen 18: development",
    ("matched sparse", "R18 development"): "Matched sparse: exploratory",
    ("no-crisis ablation", "R18 development"): "No-crisis ablation: exploratory",
    ("R15 gen18", "R18 sealed"): "R15 gen 18: sealed",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_comparison() -> list[dict]:
    records: list[dict] = []
    with COMPARISON_CSV.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            key = (raw["candidate"], raw["partition"])
            record = {
                "candidate": raw["candidate"],
                "partition": raw["partition"],
                "label": DISPLAY_LABELS[key],
                "color": ROW_COLORS[key],
                "robust_score": float(raw["robust_score"]),
                "repeat_scores": np.asarray(
                    [float(raw[f"repeat_{index}"]) for index in range(1, 4)],
                    dtype=np.float64,
                ),
                "sham_effect": float(raw["sham_effect"]),
                "injury_effect": float(raw["injury_effect"]),
                "operator_fraction": {
                    "clone": float(raw["clone"]),
                    **{operator: float(raw[operator]) for operator in OPERATORS},
                },
            }
            records.append(record)
    if len(records) != 8:
        raise ValueError(f"expected eight frozen comparison rows, found {len(records)}")
    return records


def _style_effect_axis(axis: plt.Axes) -> None:
    axis.axvline(0.0, color="#111827", linewidth=1.0, zorder=0)
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_composite(records: list[dict], output_dir: Path) -> None:
    figure = plt.figure(figsize=(15.5, 12.0), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.08, 1.0))
    repeat_axis = figure.add_subplot(grid[0, 0])
    treatment_axis = figure.add_subplot(grid[0, 1])
    rate_axis = figure.add_subplot(grid[1, 0])
    composition_axis = figure.add_subplot(grid[1, 1])

    y = np.arange(len(records))[::-1]
    for row, record in zip(y, records, strict=True):
        repeat_axis.scatter(
            record["repeat_scores"],
            np.full(3, row),
            color=record["color"],
            edgecolor="white",
            linewidth=0.6,
            s=62,
            alpha=0.88,
            zorder=3,
        )
        repeat_axis.scatter(
            record["robust_score"],
            row,
            marker="D",
            color="#111827",
            edgecolor="white",
            linewidth=0.6,
            s=54,
            zorder=4,
        )
    repeat_axis.set_yticks(y, [record["label"] for record in records])
    repeat_axis.set_xlabel("paired robust score vs exact clone")
    repeat_axis.set_title(
        "A  Replicated effects (black diamond = coherent median)",
        loc="left",
        fontweight="bold",
    )
    _style_effect_axis(repeat_axis)

    offset = 0.13
    for row, record in zip(y, records, strict=True):
        treatment_axis.plot(
            [record["sham_effect"], record["injury_effect"]],
            [row + offset, row - offset],
            color="#9CA3AF",
            linewidth=1.0,
            zorder=1,
        )
        treatment_axis.scatter(
            record["sham_effect"],
            row + offset,
            marker="o",
            color="#2A9D8F",
            s=55,
            label="sham" if row == y[0] else None,
            zorder=3,
        )
        treatment_axis.scatter(
            record["injury_effect"],
            row - offset,
            marker="^",
            color="#E76F51",
            s=62,
            label="injury" if row == y[0] else None,
            zorder=3,
        )
    treatment_axis.set_yticks(y, [record["label"] for record in records])
    treatment_axis.set_xlabel("mean AUC delta vs exact clone")
    treatment_axis.set_title(
        "B  Sham and injury effects in the selected repeat",
        loc="left",
        fontweight="bold",
    )
    treatment_axis.legend(frameon=False, loc="lower right", ncols=2)
    _style_effect_axis(treatment_axis)

    sparse = [
        record
        for record in records
        if record["candidate"] not in {"fixed standard", "fixed conservative"}
    ]
    sparse_y = np.arange(len(sparse))[::-1]
    non_clone = np.asarray(
        [100.0 * (1.0 - item["operator_fraction"]["clone"]) for item in sparse]
    )
    rate_axis.barh(
        sparse_y,
        non_clone,
        color=[item["color"] for item in sparse],
        height=0.62,
    )
    for row, value in zip(sparse_y, non_clone, strict=True):
        rate_axis.text(value + 0.07, row, f"{value:.2f}%", va="center", fontsize=9)
    rate_axis.set_yticks(sparse_y, [item["label"] for item in sparse])
    rate_axis.set_xlim(0.0, max(non_clone) * 1.18)
    rate_axis.set_xlabel("non-clone share of realized births (%)")
    rate_axis.set_title(
        "C  Mutation frequency among sparse schedulers",
        loc="left",
        fontweight="bold",
    )
    rate_axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    rate_axis.set_axisbelow(True)
    rate_axis.spines[["top", "right"]].set_visible(False)

    left = np.zeros(len(sparse), dtype=np.float64)
    for operator in OPERATORS:
        raw = np.asarray(
            [item["operator_fraction"][operator] for item in sparse],
            dtype=np.float64,
        )
        total = np.asarray(
            [1.0 - item["operator_fraction"]["clone"] for item in sparse],
            dtype=np.float64,
        )
        values = np.divide(raw, total, out=np.zeros_like(raw), where=total > 0.0)
        if not np.any(values > 0.0):
            continue
        composition_axis.barh(
            sparse_y,
            100.0 * values,
            left=left,
            height=0.62,
            color=OPERATOR_COLORS[operator],
            label=OPERATOR_LABELS[operator],
        )
        left += 100.0 * values
    composition_axis.set_yticks(sparse_y, [item["label"] for item in sparse])
    composition_axis.set_xlim(0.0, 100.0)
    composition_axis.set_xlabel("composition among non-clone births (%)")
    composition_axis.set_title(
        "D  Which mutation operators were used",
        loc="left",
        fontweight="bold",
    )
    composition_axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncols=3,
    )
    composition_axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "R15–R20: a development-only mechanism that failed sealed transfer",
        fontsize=18,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"r15_r20_results_composite.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(figure)


def _pair_records(metrics_path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    metrics = _read_json(metrics_path)
    manifest = _read_json(manifest_path)
    pairs: OrderedDict[str, dict] = OrderedDict()
    for world in manifest["worlds"]:
        pairs.setdefault(
            world["pair_id"],
            {
                "pair_id": world["pair_id"],
                "founder_id": world["founder_id"],
                "world_seed": int(world["world_seed"]),
            },
        )
    pair_deltas = metrics["private"]["pair_deltas"]
    if len(pairs) != len(pair_deltas):
        raise ValueError(
            f"{metrics_path}: {len(pair_deltas)} deltas for {len(pairs)} manifest pairs"
        )
    records = []
    for pair, delta in zip(pairs.values(), pair_deltas, strict=True):
        records.append({**pair, "pair_delta": float(delta)})
    return records, metrics


def _plot_founder_panel(
    axis: plt.Axes,
    records: list[dict],
    metrics: dict,
    *,
    title: str,
    color: str,
) -> None:
    by_founder: OrderedDict[str, list[dict]] = OrderedDict()
    for record in records:
        by_founder.setdefault(record["founder_id"], []).append(record)
    founders = list(by_founder)
    y = np.arange(len(founders))[::-1]
    seed_values = sorted({record["world_seed"] for record in records})
    seed_markers = {seed: marker for seed, marker in zip(seed_values, ("o", "s"), strict=True)}

    for row, founder in zip(y, founders, strict=True):
        values = by_founder[founder]
        xs = np.asarray([value["pair_delta"] for value in values], dtype=np.float64)
        axis.plot(xs, np.full(xs.shape, row), color="#9CA3AF", linewidth=1.2, zorder=1)
        for value in values:
            axis.scatter(
                value["pair_delta"],
                row,
                marker=seed_markers[value["world_seed"]],
                s=58,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
        axis.scatter(
            float(np.mean(xs)),
            row,
            marker="D",
            s=42,
            color="#111827",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )

    robust_score = float(metrics["combined_score"])
    axis.axvline(0.0, color="#111827", linewidth=1.0, zorder=0)
    axis.axvline(
        robust_score,
        color=color,
        linewidth=1.4,
        linestyle="--",
        zorder=0,
    )
    axis.set_yticks(y, founders)
    axis.set_title(
        f"{title}\nfrozen aggregate = {robust_score:+.6f}",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        handles=[
            *[
                Line2D(
                    [0],
                    [0],
                    marker=seed_markers[seed],
                    linestyle="none",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markersize=7,
                    label=f"world seed {seed}",
                )
                for seed in seed_values
            ],
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor="#111827",
                markeredgecolor="white",
                markersize=6,
                label="founder mean",
            ),
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="--",
                linewidth=1.4,
                label="frozen aggregate",
            ),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8.5,
    )


def _plot_founder_evidence(output_dir: Path) -> None:
    development, development_metrics = _pair_records(
        DEVELOPMENT_METRICS, DEVELOPMENT_MANIFEST
    )
    sealed, sealed_metrics = _pair_records(SEALED_METRICS, SEALED_MANIFEST)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.2),
        sharex=True,
        layout="constrained",
    )
    _plot_founder_panel(
        axes[0],
        development,
        development_metrics,
        title="A  R18 development (selected repeat)",
        color="#2563EB",
    )
    _plot_founder_panel(
        axes[1],
        sealed,
        sealed_metrics,
        title="B  R18 sealed (selected repeat)",
        color="#DC2626",
    )
    axes[0].set_xlabel("punctuated pair effect vs exact clone")
    axes[1].set_xlabel("punctuated pair effect vs exact clone")
    figure.suptitle(
        "R15 generation 18: transfer varied across founder/world pairs",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.025,
        "Each pair effect is the candidate-minus-clone difference of the sham/injury harmonic mean. "
        "Panels share the same x scale.",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"r15_r20_founder_evidence.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    records = _read_comparison()
    _plot_composite(records, output_dir)
    _plot_founder_evidence(output_dir)


if __name__ == "__main__":
    main()
