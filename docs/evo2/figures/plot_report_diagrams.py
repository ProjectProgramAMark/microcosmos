"""Generate self-contained architecture and protocol figures for the Evo² report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent

BLUE = "#2563eb"
BLUE_LIGHT = "#dbeafe"
GREEN = "#16a34a"
GREEN_LIGHT = "#dcfce7"
ORANGE = "#ea580c"
ORANGE_LIGHT = "#ffedd5"
PURPLE = "#7c3aed"
PURPLE_LIGHT = "#ede9fe"
RED = "#dc2626"
RED_LIGHT = "#fee2e2"
INK = "#111827"
MUTED = "#4b5563"
GRID = "#d1d5db"
PALE = "#f8fafc"


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def box(
    ax,
    xy,
    width,
    height,
    text,
    *,
    facecolor=PALE,
    edgecolor=GRID,
    fontsize=11,
    weight="normal",
    color=INK,
    radius=0.02,
    linewidth=1.5,
    zorder=2,
):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        zorder=zorder + 1,
    )
    return patch


def arrow(ax, start, end, *, color=MUTED, width=1.8, style="-|>", zorder=3):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        shrinkA=5,
        shrinkB=5,
        linewidth=width,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def save(fig, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.png", dpi=220)
    fig.savefig(HERE / f"{stem}.pdf")
    plt.close(fig)


def rsi_loop() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Evo² nested improvement loop", fontsize=20, fontweight="bold", pad=16)

    stages = [
        (0.84, "ShinkaEvolve proposes a\nheredity-policy program", BLUE_LIGHT, BLUE),
        (0.71, "Policy scores six trusted\nheredity actions", BLUE_LIGHT, BLUE),
        (0.58, "TensorNEAT applies the action\nto produce a child CPPN", "#e7efe5", "#6b8e67"),
        (0.45, "The CPPN controls an\nembodied filament", "#e7efe5", "#6b8e67"),
        (0.32, "Microcosmos ecology: move, feed,\nreproduce, and die", "#e7efe5", "#6b8e67"),
        (0.19, "The frozen evaluator scores the\npopulation against exact clone", BLUE_LIGHT, BLUE),
    ]
    x, width, height = 0.08, 0.64, 0.085
    for index, (y, label, fill, edge) in enumerate(stages):
        box(
            ax,
            (x, y),
            width,
            height,
            label,
            facecolor=fill,
            edgecolor=edge,
            fontsize=11.5,
        )
        if index < len(stages) - 1:
            next_y = stages[index + 1][0]
            arrow(
                ax,
                (x + width / 2, y),
                (x + width / 2, next_y + height),
                color=INK,
                width=1.6,
            )

    feedback = FancyArrowPatch(
        (x + width, 0.19 + height / 2),
        (x + width, 0.84 + height / 2),
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.43",
        mutation_scale=14,
        shrinkA=5,
        shrinkB=5,
        linewidth=2.0,
        color="#5678c8",
        zorder=2,
    )
    ax.add_patch(feedback)
    ax.text(
        0.87,
        0.53,
        "results guide\nthe next proposal",
        ha="left",
        va="center",
        fontsize=11,
        fontstyle="italic",
        color="#5678c8",
    )

    boundary = FancyBboxPatch(
        (0.08, 0.035),
        0.64,
        0.105,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor="#f3f4f6",
        edgecolor="#9ca3af",
        linestyle=(0, (5, 3)),
        linewidth=1.6,
    )
    ax.add_patch(boundary)
    ax.text(
        0.40,
        0.105,
        "Fixed boundary: Shinka cannot edit any of this",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.40,
        0.063,
        "physics · resources · injury · mutation kernels · founders\n"
        "score · seeds · budget · hidden holdouts",
        ha="center",
        va="center",
        fontsize=9.4,
        color=INK,
    )

    save(fig, "rsi_loop")


def implementation_design() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    fig.suptitle(
        "Two design choices keep an evolving ecosystem GPU-native",
        fontsize=21,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0]
    ax.set_title("A  Dynamic life cycles in static organism slots", loc="left", fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.90, "Fixed capacity: 32 slots (8 shown)", fontweight="bold", color=INK)
    x0, y0, gap, w, h = 0.04, 0.75, 0.012, 0.105, 0.11
    for i in range(8):
        alive = i < 5
        rect = FancyBboxPatch(
            (x0 + i * (w + gap), y0),
            w,
            h,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor=GREEN_LIGHT if alive else "white",
            edgecolor=GREEN if alive else GRID,
            linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + i * (w + gap) + w / 2,
            y0 + h / 2,
            f"{i}\n{'alive' if alive else 'free'}",
            ha="center",
            va="center",
            fontsize=9.5,
            color=INK if alive else MUTED,
        )

    arrow(ax, (0.19, 0.70), (0.19, 0.61), color=RED)
    ax.text(0.205, 0.655, "death: mask off", color=RED, fontsize=9.5, va="center")
    arrow(ax, (0.77, 0.61), (0.77, 0.70), color=GREEN)
    ax.text(0.785, 0.655, "birth: reuse slot", color=GREEN, fontsize=9.5, va="center")

    box(
        ax,
        (0.05, 0.45),
        0.90,
        0.10,
        "alive[slot]  →  node / edge / hinge activity masks",
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        fontsize=12,
        weight="bold",
    )
    for x, label in zip(
        (0.05, 0.285, 0.52, 0.755),
        ("constraints", "steric + fluid", "resources", "controller"),
    ):
        box(ax, (x, 0.27), 0.19, 0.09, label, facecolor="white", edgecolor=GRID, fontsize=10)
        arrow(ax, (0.50, 0.45), (x + 0.095, 0.36), color=BLUE, width=1.2)

    box(
        ax,
        (0.07, 0.08),
        0.86,
        0.10,
        "Biology changes occupancy and state,\nnot array shape",
        facecolor=GREEN_LIGHT,
        edgecolor=GREEN,
        fontsize=11.5,
        weight="bold",
        color="#14532d",
    )

    ax = axes[1]
    ax.set_title("B  Variable CPPNs in padded fixed-size tensors", loc="left", fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.04, 0.73), 0.20, 0.15, "CPPN genome", facecolor=PURPLE_LIGHT, edgecolor=PURPLE, weight="bold")
    ax.text(0.14, 0.70, "inherited per organism", ha="center", fontsize=9.5, color=MUTED)

    # Draw compact tensor blocks rather than a literal dense matrix.
    ax.add_patch(Rectangle((0.32, 0.73), 0.20, 0.15, facecolor="white", edgecolor=PURPLE, linewidth=1.5))
    ax.add_patch(Rectangle((0.32, 0.79), 0.20, 0.09, facecolor=PURPLE_LIGHT, edgecolor="none"))
    ax.text(0.42, 0.835, "used nodes", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0.42, 0.76, "padding", ha="center", va="center", fontsize=9.5, color=MUTED)
    ax.text(0.42, 0.70, "15 × 5 node tensor", ha="center", fontsize=9.5, color=MUTED)

    ax.add_patch(Rectangle((0.60, 0.73), 0.20, 0.15, facecolor="white", edgecolor=PURPLE, linewidth=1.5))
    ax.add_patch(Rectangle((0.60, 0.78), 0.20, 0.10, facecolor=PURPLE_LIGHT, edgecolor="none"))
    ax.text(0.70, 0.83, "used edges", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0.70, 0.755, "padding", ha="center", va="center", fontsize=9.5, color=MUTED)
    ax.text(0.70, 0.70, "30-connection capacity", ha="center", fontsize=9.5, color=MUTED)
    arrow(ax, (0.24, 0.805), (0.31, 0.805), color=PURPLE)
    arrow(ax, (0.53, 0.805), (0.59, 0.805), color=PURPLE)

    box(
        ax,
        (0.03, 0.48),
        0.32,
        0.13,
        "position · time\nlocal resource\nhead–tail resource gradient",
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        fontsize=9.5,
    )
    box(ax, (0.42, 0.49), 0.21, 0.11, "jax.vmap\nover organisms", facecolor=BLUE_LIGHT, edgecolor=BLUE, weight="bold")
    box(ax, (0.71, 0.49), 0.24, 0.11, "bounded bending\ncommands", facecolor=GREEN_LIGHT, edgecolor=GREEN, weight="bold")
    arrow(ax, (0.35, 0.545), (0.41, 0.545), color=BLUE)
    arrow(ax, (0.63, 0.545), (0.70, 0.545), color=BLUE)

    box(
        ax,
        (0.11, 0.25),
        0.78,
        0.11,
        "Trusted TensorNEAT mutation changes\nparameters or topology at birth",
        facecolor=PURPLE_LIGHT,
        edgecolor=PURPLE,
        fontsize=11,
        weight="bold",
    )
    arrow(ax, (0.50, 0.36), (0.50, 0.48), color=PURPLE)
    box(
        ax,
        (0.07, 0.08),
        0.86,
        0.10,
        "Variable graphs, fixed shapes,\nand batched GPU inference",
        facecolor=GREEN_LIGHT,
        edgecolor=GREEN,
        fontsize=11.5,
        weight="bold",
        color="#14532d",
    )

    fig.subplots_adjust(top=0.88, wspace=0.10)
    save(fig, "implementation_design")


def evaluation_protocol() -> None:
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Frozen evaluation: paired causal worlds and one-way evidence access",
        fontsize=21,
        fontweight="bold",
        pad=20,
    )

    ax.text(0.04, 0.91, "1  Within each policy evaluation", fontsize=14, fontweight="bold")
    box(ax, (0.04, 0.75), 0.18, 0.10, "founder + world seed\npre-event rollout", facecolor=BLUE_LIGHT, edgecolor=BLUE, weight="bold")
    box(ax, (0.30, 0.80), 0.16, 0.08, "sham", facecolor=GREEN_LIGHT, edgecolor=GREEN, weight="bold")
    box(ax, (0.30, 0.67), 0.16, 0.08, "head injury", facecolor=RED_LIGHT, edgecolor=RED, weight="bold")
    arrow(ax, (0.22, 0.80), (0.29, 0.84), color=GREEN)
    arrow(ax, (0.22, 0.80), (0.29, 0.71), color=RED)
    ax.text(
        0.04,
        0.65,
        "Sham and injury fork from one exact pre-event state.\nCandidate and clone are evaluated separately on the same founders/worlds.",
        fontsize=10.5,
        color=MUTED,
        va="top",
    )

    ax.text(0.55, 0.91, "2  Candidate versus exact clone", fontsize=14, fontweight="bold")
    box(ax, (0.55, 0.75), 0.17, 0.10, "candidate\nH(sham, injury)", facecolor=PURPLE_LIGHT, edgecolor=PURPLE, weight="bold")
    box(ax, (0.79, 0.75), 0.17, 0.10, "exact clone\nH(sham, injury)", facecolor=PALE, edgecolor=MUTED, weight="bold")
    ax.text(0.755, 0.80, "−", fontsize=28, ha="center", va="center", color=INK)
    box(
        ax,
        (0.62, 0.59),
        0.27,
        0.09,
        "pair effect",
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        weight="bold",
    )
    arrow(ax, (0.64, 0.75), (0.70, 0.68), color=ORANGE)
    arrow(ax, (0.875, 0.75), (0.81, 0.68), color=ORANGE)
    ax.text(
        0.755,
        0.54,
        "Frozen aggregate = 80% trimmed central mean\n+ 20% bottom-quartile mean across pair effects",
        ha="center",
        va="top",
        fontsize=10.5,
        color=MUTED,
    )

    ax.plot([0.04, 0.96], [0.46, 0.46], color=GRID, linewidth=1.5)
    ax.text(0.04, 0.40, "3  One-way evidence gate", fontsize=14, fontweight="bold")

    box(ax, (0.05, 0.19), 0.22, 0.13, "TRAINING\nguides Shinka search", facecolor=BLUE_LIGHT, edgecolor=BLUE, fontsize=12, weight="bold")
    box(ax, (0.39, 0.19), 0.22, 0.13, "DEVELOPMENT\none frozen qualifier", facecolor=ORANGE_LIGHT, edgecolor=ORANGE, fontsize=12, weight="bold")
    box(ax, (0.73, 0.19), 0.22, 0.13, "SEALED\none candidate · one use", facecolor=RED_LIGHT, edgecolor=RED, fontsize=12, weight="bold")
    arrow(ax, (0.27, 0.255), (0.38, 0.255), color=ORANGE, width=2.2)
    arrow(ax, (0.61, 0.255), (0.72, 0.255), color=RED, width=2.2)
    ax.text(
        0.325,
        0.34,
        "positive replicated score\n+ non-clone ecology logic",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )
    ax.text(
        0.665,
        0.34,
        "all 3 development\nrepeats positive",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )
    ax.text(
        0.50,
        0.08,
        "Three complete GPU executions measure numerical stability; founders/worlds, not reruns, are the ecological units.",
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    save(fig, "evaluation_protocol")


def campaign_timeline() -> None:
    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Campaign progression: each redesign followed a measured failure",
        fontsize=21,
        fontweight="bold",
        pad=18,
    )

    xs = [0.04, 0.20, 0.36, 0.52, 0.68, 0.84]
    colors = [BLUE, BLUE, PURPLE, ORANGE, RED, GREEN]
    fills = [BLUE_LIGHT, BLUE_LIGHT, PURPLE_LIGHT, ORANGE_LIGHT, RED_LIGHT, GREEN_LIGHT]
    titles = [
        "R1–R6",
        "R7–R14",
        "R15",
        "R16b–R17b",
        "R18–R20",
        "R21",
    ]
    bodies = [
        "Build lifecycle,\noperator ABI, and\nreal stressor",
        "Correct evaluator,\nABI, and head-\ninjury benchmark",
        "Shinka discovers\nsparse ecology-\nconditioned policy",
        "Development wins;\nindependent sealed\nfailures",
        "Expose one-repeat\nnoise; require three\nfull reruns",
        "Final frozen\ncross-founder\ntransfer test",
    ]
    y = 0.43
    for i, (x, color, fill, title, body) in enumerate(zip(xs, colors, fills, titles, bodies)):
        box(ax, (x, y), 0.125, 0.28, "", facecolor=fill, edgecolor=color, linewidth=2)
        ax.text(x + 0.0625, y + 0.22, title, ha="center", va="center", fontsize=13, fontweight="bold", color=color)
        ax.text(x + 0.0625, y + 0.11, body, ha="center", va="center", fontsize=9.7, color=INK, linespacing=1.25)
        if i < len(xs) - 1:
            arrow(ax, (x + 0.128, y + 0.14), (xs[i + 1] - 0.003, y + 0.14), color=MUTED, width=1.8)

    ax.text(
        0.50,
        0.25,
        "Implementation and experimental design changed between frozen rounds; Shinka generated programs inside each round.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    box(
        ax,
        (0.20, 0.08),
        0.60,
        0.095,
        "No R22: R21 closes the research phase, regardless of sign",
        facecolor=PALE,
        edgecolor=INK,
        fontsize=11.5,
        weight="bold",
    )

    save(fig, "campaign_timeline")


def main() -> None:
    setup()
    rsi_loop()
    implementation_design()
    evaluation_protocol()
    campaign_timeline()


if __name__ == "__main__":
    main()
