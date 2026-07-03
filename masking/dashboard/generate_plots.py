"""Generate summary plots for masking experiments A/B/C from result JSONL files.

Usage:
    python3 masking/dashboard/generate_plots.py \
        --a batch_experiment_a.jsonl --b batch_experiment_b.jsonl --c batch_experiment_c.jsonl \
        --outdir masking/dashboard/plots
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# Validated categorical/sequential palette (light mode) -- see dataviz skill references/palette.md
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQ_BLUE = "#2a78d6"
SLOT_1 = "#2a78d6"  # blue -- prefix
SLOT_2 = "#1baf7a"  # aqua -- suffix
DIVERGE_BLUE = "#2a78d6"  # baseline / full reasoning
DIVERGE_RED = "#e34948"   # masked / perturbed condition

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _load(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def plot_experiment_a(rows: list[dict], outpath: Path) -> None:
    by_cluster = defaultdict(list)
    for r in rows:
        by_cluster[r["event_cluster"]].append(r["ade_m"])
    items = sorted(by_cluster.items(), key=lambda kv: statistics.mean(kv[1]))
    labels = [k.replace("_", " ").title() for k, _ in items]
    means = [statistics.mean(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    bars = ax.barh(labels, means, color=SEQ_BLUE, height=0.6, zorder=3)
    for bar, val, (_, v) in zip(bars, means, items):
        ax.text(val + max(means) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f} m  (n={len(v)})", va="center", ha="left",
                 fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("Mean ADE when reasoning is masked (m)")
    ax.set_title("Experiment A — ADE from masking CoT reasoning, by scenario",
                 fontsize=13, color=INK_PRIMARY, loc="left", pad=12)
    ax.set_xlim(0, max(means) * 1.35)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_experiment_b(rows: list[dict], outpath: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: most impactful recurring words (per-word leave-one-out salience)
    word_ades = defaultdict(list)
    for r in rows:
        for w in r["per_word_salience_top20"]:
            word_ades[w["word"].lower().strip(".,")].append(w["traj_ade_m"])
    multi = {w: v for w, v in word_ades.items() if len(v) >= 3}
    ranked = sorted(multi.items(), key=lambda kv: statistics.mean(kv[1]))[-10:]
    labels1 = [w for w, _ in ranked]
    means1 = [statistics.mean(v) for _, v in ranked]
    bars1 = ax1.barh(labels1, means1, color=SLOT_1, height=0.6, zorder=3)
    for bar, val in zip(bars1, means1):
        ax1.text(val + max(means1) * 0.02, bar.get_y() + bar.get_height() / 2,
                  f"{val:.3f} m", va="center", ha="left", fontsize=9, color=INK_SECONDARY)
    ax1.set_xlabel("Mean ADE when word is removed (m)")
    ax1.set_title("Most impactful recurring words", fontsize=11, color=INK_PRIMARY, loc="left")
    ax1.set_xlim(0, max(means1) * 1.35)
    ax1.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax1.spines[spine].set_visible(False)
    ax1.spines["bottom"].set_color(BASELINE)

    # Right: concept-set ablation ADE by scenario
    by_cluster = defaultdict(list)
    for r in rows:
        by_cluster[r["event_cluster"]].append(r["ade_m"])
    items = sorted(by_cluster.items(), key=lambda kv: statistics.mean(kv[1]))
    labels2 = [k.replace("_", " ").title() for k, _ in items]
    means2 = [statistics.mean(v) for _, v in items]
    bars2 = ax2.barh(labels2, means2, color=SLOT_2, height=0.6, zorder=3)
    for bar, val in zip(bars2, means2):
        ax2.text(val + max(means2) * 0.03, bar.get_y() + bar.get_height() / 2,
                  f"{val:.3f} m", va="center", ha="left", fontsize=9, color=INK_SECONDARY)
    ax2.set_xlabel("Mean ADE from concept-word ablation (m)")
    ax2.set_title("Concept-set ablation, by scenario", fontsize=11, color=INK_PRIMARY, loc="left")
    ax2.set_xlim(0, max(means2) * 1.35 if max(means2) > 0 else 1)
    ax2.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.spines["bottom"].set_color(BASELINE)

    fig.suptitle("Experiment B — per-word salience & concept ablation",
                 fontsize=12, color=INK_PRIMARY, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_experiment_c(rows: list[dict], outpath: Path) -> None:
    prefix_by_n = defaultdict(list)
    suffix_by_n = defaultdict(list)
    for r in rows:
        for entry in r["prefix_sweep"]:
            prefix_by_n[entry["n"]].append(entry["ade_m"])
        for entry in r["suffix_sweep"]:
            suffix_by_n[entry["n"]].append(entry["ade_m"])

    ns = sorted(prefix_by_n)
    prefix_means = [statistics.mean(prefix_by_n[n]) for n in ns]
    suffix_means = [statistics.mean(suffix_by_n[n]) for n in ns]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ns, prefix_means, marker="o", markersize=6, linewidth=2,
            color=SLOT_1, label="Prefix (sees first N words)", zorder=3)
    ax.plot(ns, suffix_means, marker="o", markersize=6, linewidth=2,
            color=SLOT_2, label="Suffix (sees words after N)", zorder=3)
    for n, v in zip(ns, prefix_means):
        ax.annotate(f"{v:.3f}", (n, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=INK_SECONDARY)
    for n, v in zip(ns, suffix_means):
        ax.annotate(f"{v:.3f}", (n, v), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color=INK_SECONDARY)

    ax.set_xlabel("Word-count threshold N")
    ax.set_ylabel("Mean ADE vs. full-reasoning baseline (m)")
    ax.set_title("Experiment C — where trajectory-relevant information sits in the CoT",
                 fontsize=12, color=INK_PRIMARY, loc="left", pad=12)
    ax.set_ylim(0, max(prefix_means + suffix_means) * 1.28)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    legend = ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_trajectory_examples(rows: list[dict], outpath: Path, n_top: int = 3, n_bottom: int = 1) -> None:
    ranked = sorted(rows, key=lambda r: -r["ade_m"])
    picks = ranked[:n_top] + ranked[-n_bottom:]

    n = len(picks)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 4.4 * nrows))
    axes = axes.flatten()

    for ax, r in zip(axes, picks):
        none_xy = list(zip(*r["traj_none_xy"]))
        masked_xy = list(zip(*r["traj_masked_xy"]))
        ax.plot(none_xy[0], none_xy[1], color=DIVERGE_BLUE, linewidth=2.5,
                marker="o", markersize=3, label="Full reasoning", zorder=3)
        ax.plot(masked_xy[0], masked_xy[1], color=DIVERGE_RED, linewidth=2.5,
                marker="o", markersize=3, label="Reasoning masked", zorder=3,
                linestyle="--")
        ax.scatter([0], [0], color=INK_PRIMARY, s=30, zorder=4)
        ax.set_title(f"{r['clip_id'][:8]} · {r['event_cluster'].replace('_', ' ').title()}\n"
                     f"ADE={r['ade_m']:.3f} m  endpoint={r['endpoint_m']:.3f} m",
                     fontsize=9.5, color=INK_PRIMARY, loc="left")
        # Independent (non-equal) axis scaling -- lateral divergence is usually
        # small relative to forward travel distance, so a geometrically "honest"
        # equal-aspect plot squashes the very divergence this chart exists to show.
        ax.set_xlabel("Forward (m)", fontsize=8, color=INK_MUTED)
        ax.set_ylabel("Lateral (m)", fontsize=8, color=INK_MUTED)
        ax.grid(color=GRIDLINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(BASELINE)
        ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Experiment A — example trajectories, full reasoning vs. masked",
                 fontsize=13, color=INK_PRIMARY, y=1.08, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c", required=True)
    ap.add_argument("--outdir", default="masking/dashboard/plots")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows_a = _load(args.a)
    plot_experiment_a(rows_a, outdir / "experiment_a.png")
    plot_experiment_b(_load(args.b), outdir / "experiment_b.png")
    plot_experiment_c(_load(args.c), outdir / "experiment_c.png")
    plot_trajectory_examples(rows_a, outdir / "experiment_a_trajectories.png")
    print(f"Wrote plots to {outdir}/")


if __name__ == "__main__":
    main()
