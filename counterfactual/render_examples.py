# SPDX-License-Identifier: Apache-2.0
"""
render_examples.py — top-down (x, y) trajectory comparison plots for a
small, curated set of counterfactual examples (see EXAMPLES below), one PNG
per (scene_id, step, token): baseline path vs. Option A's (isolated swap)
path vs. Option B's (forced + re-sampled) path, overlaid.

Requires counterfactual/configs/examples.yaml to have been run with
capture_trajectories=true and fetched via
`python -m counterfactual.fetch_from_logs --workload_id <id>
--out_dir counterfactual/results/examples` -- deliberately a SEPARATE
directory from counterfactual/results/ (the full sweep's data, no
trajectories captured there) so a rerun of this small curated set can never
silently overwrite the primary aggregate-stats dataset, even though the
scene_ids overlap and the underlying generation is seeded to reproduce
identically.

Runs locally (pandas/matplotlib, no model/GPU needed) -- this module only
reads the already-fetched JSON and plots.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from counterfactual.report import example_key, is_degenerate

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# (scene_id, step, token) -- the specific examples surfaced during review:
# the "Nudge left..." illustration, the top-8 clean Option B effects, and
# the low-probability tokenizer-fragment case (biggest Option A-only effect).
EXAMPLES: list[tuple[str, int, str]] = [
    ("2e1d7d7c-8ce5-4ac0-b13d-0dad5fd34a1f_9981188", 0, "N"),
    ("2e1d7d7c-8ce5-4ac0-b13d-0dad5fd34a1f_9981188", 38, "right"),
    ("71dc37e6-f69e-43a8-8d6b-e472f6b92d7a_4875293", 0, "Change"),
    ("625c3a0d-3273-4dff-a6f7-1cae79a93436_4994763", 0, "Change"),
    ("065131ea-2a04-4f34-bb1b-22543be873a9_1577426", 0, "Change"),
    ("a994de45-2c85-45dc-b45d-11964806fb44_16693431", 0, "N"),
    ("14529dca-ea8e-4f4c-81a6-cbe365dfde7d_9144272", 0, "Stop"),
    ("a994de45-2c85-45dc-b45d-11964806fb44_16693431", 1, "ain"),
]

# Same validated two-series palette as pref_pairs/noise_floor_report.py's
# Compare tab (--cmp-a/--cmp-b) -- light-mode values, since these PNGs are
# static raster images (no dark-mode variant) embedded on a light page area.
_COLOR_BASELINE = "#565C63"  # matches the report's --ink-dim
_COLOR_A = "#2a78d6"
_COLOR_B = "#eb6834"


def _find_alt(positions: list[dict], step: int, token: str) -> dict | None:
    pos = next((p for p in positions if p["step"] == step), None)
    if pos is None:
        return None
    return next((a for a in pos["alternatives"] if a["token"].strip() == token.strip()), None)


def render_one(scene_json: dict[str, Any], step: int, token: str, out_path: Path) -> bool:
    """Renders one comparison PNG. Returns False (does not raise) if the
    example isn't found in this scene's data or has no captured trajectory
    -- a missing curated example shouldn't take down the whole batch."""
    baseline_xy = scene_json.get("baseline_xy_a") or scene_json.get("baseline_xy_b")
    alt_a = _find_alt(scene_json["single_token_swap_sweep"], step, token)
    alt_b = _find_alt(scene_json["counterfactual_sweep"], step, token)

    if baseline_xy is None or alt_a is None or alt_a.get("xy") is None:
        logger.warning("missing baseline or Option A trajectory for step=%d token=%r -- skipping", step, token)
        return False

    fig, ax = plt.subplots(figsize=(5, 5))
    base = np.asarray(baseline_xy)
    ax.plot(base[:, 0], base[:, 1], color=_COLOR_BASELINE, linewidth=2, label="baseline", zorder=3)

    a_xy = np.asarray(alt_a["xy"])
    ax.plot(a_xy[:, 0], a_xy[:, 1], color=_COLOR_A, linewidth=1.6, linestyle="--",
            label=f"Option A (ade={alt_a['traj_ade_m']:.3f}m)", zorder=2)

    if alt_b is not None and alt_b.get("xy") is not None:
        if not is_degenerate(alt_b):
            b_xy = np.asarray(alt_b["xy"])
            ax.plot(b_xy[:, 0], b_xy[:, 1], color=_COLOR_B, linewidth=1.6, linestyle="--",
                    label=f"Option B (ade={alt_b['traj_ade_m']:.3f}m)", zorder=2)

    ax.scatter([0], [0], marker="*", color="black", s=90, zorder=5, label="t=0 (ego)")
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=7)
    ax.set_title(f"step {step}: swap to '{token.strip()}'", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def render_all(results_dir: str | Path, out_dir: str | Path) -> list[str]:
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    written = []
    for scene_id, step, token in EXAMPLES:
        path = results_dir / f"{scene_id}.json"
        if not path.exists():
            logger.warning("no results file for scene %s -- skipping its examples", scene_id)
            continue
        scene_json = json.loads(path.read_text())
        key = example_key(scene_id, step, token)
        if render_one(scene_json, step, token, out_dir / f"{key}.png"):
            written.append(key)
    return written


def load_example_plots_b64(plot_dir: str | Path) -> dict[str, str]:
    """Reads every {key}.png in plot_dir and base64-encodes it, keyed by
    filename stem -- same load-and-key pattern as
    pref_pairs.noise_floor_report.load_images_b64."""
    plot_dir = Path(plot_dir)
    if not plot_dir.is_dir():
        return {}
    return {p.stem: base64.b64encode(p.read_bytes()).decode("ascii") for p in sorted(plot_dir.glob("*.png"))}


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="counterfactual/results/examples")
    ap.add_argument("--out_dir", default="counterfactual/results/example_plots")
    args = ap.parse_args()

    written = render_all(args.results_dir, args.out_dir)
    logger.info("wrote %d/%d example plot(s) to %s", len(written), len(EXAMPLES), args.out_dir)


if __name__ == "__main__":
    main()
