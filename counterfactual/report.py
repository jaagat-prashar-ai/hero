# SPDX-License-Identifier: Apache-2.0
"""
report.py — consolidates counterfactual/results/*.json (one file per scene,
each holding token_alternative_map + single_token_swap_sweep [Option A] +
counterfactual_sweep [Option B], see fetch_from_logs.py) into one HTML
fragment: aggregate stats + a per-scene drill-down of every swapped
position's trajectory deltas. Meant to be embedded as a tab in
pref_pairs/noise_floor_report.py's page (see that module's render_html
counterfactual_html param), not published standalone -- this module has no
_PAGE_TEMPLATE of its own.

DEGENERATE GENERATION, and why every function here treats it explicitly:
Option B (counterfactual_sweep) forces one token then lets the VLM re-sample
the rest of the reasoning. Confirmed against the real 123-scene run: 55/6944
(0.8%) of those forced continuations never emit a closing <|cot_end|>
within the generation budget -- forced_cot["cot"] comes back as "". Those 55
have a mean traj_ade_m of 5.92m vs 0.24m for the 6889 that closed cleanly
(25x) -- the diffusion expert is decoding from an incomplete/degenerate KV
state, not from a coherent alternate plan, so these numbers do not measure
"reasoning sensitivity" and must not be mixed into any ranking or aggregate
that claims to. is_degenerate() below is the single place that decision is
made; every aggregate/ranking function filters through it explicitly rather
than silently including or silently dropping these rows.
"""

from __future__ import annotations

import html
import json
import statistics
from pathlib import Path
from typing import Any

_PALETTE_A = "cmp-a"  # Option A (isolated single-token swap)
_PALETTE_B = "cmp-b"  # Option B (forced token + coherent re-sample)


def _esc(v: Any) -> str:
    return html.escape(str(v))


def _fmt(v: float, sig: int = 3) -> str:
    return "0" if v == 0 else f"{v:.{sig}g}"


def is_degenerate(alt: dict[str, Any]) -> bool:
    """True if this Option B alternative's forced generation never closed
    its <|cot_end|> span -- see module docstring. Option A alternatives are
    never degenerate in this sense (they don't regenerate text at all)."""
    cot = alt.get("forced_cot")
    if not isinstance(cot, dict):
        return False
    text = (cot.get("cot") or [""])[0]
    return text.strip() == ""


def build_counterfactual_data(results_dir: str | Path) -> dict[str, Any]:
    """Load every counterfactual/results/{scene_id}.json and return the
    structure render_counterfactual_section renders. Raises if results_dir
    has no scene files -- an empty tab would be a silent, confusing gap."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise ValueError(f"no scene JSON files found in {results_dir}")

    scenes = []
    clean_a: list[float] = []
    clean_b: list[float] = []
    clean_b_step0: list[float] = []
    clean_b_other: list[float] = []
    n_degenerate = 0
    n_alts_total = 0

    for path in files:
        scene_id = path.stem
        row = json.loads(path.read_text())
        cot_text = (row["token_alternative_map"]["cot"].get("cot") or [""])[0]
        positions_a = {p["step"]: p for p in row["single_token_swap_sweep"]}
        positions_b = {p["step"]: p for p in row["counterfactual_sweep"]}

        position_entries = []
        scene_max_clean_ade = 0.0
        for step in sorted(positions_a):
            pa = positions_a[step]
            pb = positions_b.get(step)
            alt_entries = []
            for alt_a in pa["alternatives"]:
                n_alts_total += 1
                clean_a.append(alt_a["traj_ade_m"])
                alt_b = None
                if pb:
                    alt_b = next((a for a in pb["alternatives"] if a["token"] == alt_a["token"]), None)
                degenerate = is_degenerate(alt_b) if alt_b else False
                if alt_b and not degenerate:
                    clean_b.append(alt_b["traj_ade_m"])
                    (clean_b_step0 if step == 0 else clean_b_other).append(alt_b["traj_ade_m"])
                    scene_max_clean_ade = max(scene_max_clean_ade, alt_a["traj_ade_m"], alt_b["traj_ade_m"])
                elif degenerate:
                    n_degenerate += 1
                    scene_max_clean_ade = max(scene_max_clean_ade, alt_a["traj_ade_m"])
                else:
                    scene_max_clean_ade = max(scene_max_clean_ade, alt_a["traj_ade_m"])
                alt_entries.append({
                    "token": alt_a["token"],
                    "prob": alt_a["token_prob"],
                    "ade_a": alt_a["traj_ade_m"],
                    "endpoint_a": alt_a["endpoint_shift_m"],
                    "ade_b": alt_b["traj_ade_m"] if alt_b else None,
                    "endpoint_b": alt_b["endpoint_shift_m"] if alt_b else None,
                    "forced_cot": (alt_b["forced_cot"].get("cot") or [""])[0] if alt_b else None,
                    "degenerate": degenerate,
                })
            position_entries.append({
                "step": step,
                "sampled_token": pa["sampled_token"],
                "sampled_prob": pa["sampled_prob"],
                "entropy": pa["entropy"],
                "alternatives": alt_entries,
            })

        scenes.append({
            "scene_id": scene_id,
            "cot": cot_text,
            "n_positions": len(position_entries),
            "max_clean_ade": scene_max_clean_ade,
            "positions": position_entries,
        })

    scenes.sort(key=lambda s: -s["max_clean_ade"])

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
        s = sorted(vals)
        return {
            "mean": statistics.mean(vals), "median": statistics.median(vals),
            "p90": s[int(0.9 * (len(s) - 1))], "max": s[-1],
        }

    return {
        "scenes": scenes,
        "n_scenes": len(scenes),
        "n_positions": sum(s["n_positions"] for s in scenes),
        "n_alternatives": n_alts_total,
        "n_degenerate": n_degenerate,
        "stats_a": _stats(clean_a),
        "stats_b": _stats(clean_b),
        "stats_b_step0": _stats(clean_b_step0),
        "stats_b_other": _stats(clean_b_other),
    }


def _dumbbell_row(label: str, val_a: float, val_b: float, scale_max: float) -> str:
    def pct(x: float) -> float:
        return max(0.0, min(100.0, 100.0 * x / scale_max)) if scale_max else 0.0
    xa, xb = pct(val_a), pct(val_b)
    lo, hi = min(xa, xb), max(xa, xb)
    return (
        f'<div class="dumbbell-row"><span class="dumbbell-label">{_esc(label)}</span>'
        f'<div class="dumbbell-track">'
        f'<div class="dumbbell-line" style="left:{lo:.1f}%;width:{(hi - lo):.1f}%"></div>'
        f'<div class="dumbbell-dot dot-a" style="left:{xa:.1f}%" title="Option A median: {_fmt(val_a)} m"></div>'
        f'<div class="dumbbell-dot dot-b" style="left:{xb:.1f}%" title="Option B median: {_fmt(val_b)} m"></div>'
        f'</div>'
        f'<span class="dumbbell-vals"><i class="swatch swatch-a"></i>{_fmt(val_a)}'
        f'<i class="swatch swatch-b"></i>{_fmt(val_b)}<i class="dumbbell-unit">m ADE</i></span>'
        f'</div>'
    )


def render_counterfactual_section(data: dict[str, Any]) -> str:
    """Render the full 'Token Sensitivity' tab body (everything inside the
    <div data-tab-panel="...">, not the page shell)."""
    sa, sb = data["stats_a"], data["stats_b"]
    sb0, sbo = data["stats_b_step0"], data["stats_b_other"]

    stat_tiles = "".join(
        f'<div class="masthead-stat"><b>{_esc(v)}</b><span>{_esc(label)}</span></div>'
        for v, label in [
            (data["n_scenes"], "scenes"), (data["n_positions"], "positions swept"),
            (data["n_alternatives"], "alternatives tested"), (data["n_degenerate"], "degenerate (excluded)"),
        ]
    )

    scale_max = max(sa["p90"], sb["p90"], sb0["p90"], sbo["p90"], 0.001) * 1.15
    compare_html = (
        f'<div class="compare-legend">'
        f'<span class="legend-chip"><i class="swatch swatch-a"></i>Option A (isolated single-token swap)</span>'
        f'<span class="legend-chip"><i class="swatch swatch-b"></i>Option B (forced token, reasoning re-sampled)</span>'
        f'</div>'
        f'<div class="compare-grid"><article class="compare-panel">'
        f'<h3>Trajectory delta by swap mode<span class="compare-metric-note">median ADE, meters</span></h3>'
        f'{_dumbbell_row("Overall", sa["median"], sb["median"], scale_max)}'
        f'{_dumbbell_row("First reasoning token (step 0)", sa["median"], sb0["median"], scale_max)}'
        f'{_dumbbell_row("Later reasoning tokens", sa["median"], sbo["median"], scale_max)}'
        f'</article></div>'
        f'<p class="compare-missing-note">'
        f'Full distributions (m, ADE, clean/non-degenerate only) &mdash; '
        f'Option A: mean {_fmt(sa["mean"])} &middot; p90 {_fmt(sa["p90"])} &middot; max {_fmt(sa["max"])}. '
        f'Option B: mean {_fmt(sb["mean"])} &middot; p90 {_fmt(sb["p90"])} &middot; max {_fmt(sb["max"])}. '
        f'{data["n_degenerate"]} of {data["n_alternatives"]} Option B continuations never closed their reasoning '
        f'cleanly within the generation budget and are excluded from every statistic above (mean ADE among '
        f'those degenerate ones alone: ~25&times; higher than clean ones &mdash; a KV-cache/diffusion artifact '
        f'of incomplete generation, not a reasoning-sensitivity signal).'
        f'</p>'
    )

    def alt_row(alt: dict, sampled_prob_unused: float) -> str:
        degenerate_flag = '<span class="incomplete-flag">generation incomplete</span>' if alt["degenerate"] else ""
        ade_b_html = (
            f'<span class="stat-val">{_fmt(alt["ade_b"])}<i>m</i></span>' if alt["ade_b"] is not None and not alt["degenerate"]
            else '<span class="stat-val" style="color:var(--ink-dim)">&mdash;</span>'
        )
        cot_html = ""
        if alt["forced_cot"] and not alt["degenerate"]:
            cot_html = f'<li><q>{_esc(alt["forced_cot"])}</q></li>'
        return (
            f'<div class="cf-alt-row">'
            f'<span class="cf-alt-token">&rarr; &lsquo;{_esc(alt["token"].strip())}&rsquo;</span>'
            f'<span class="cf-alt-prob">p={_fmt(alt["prob"], 2)}</span>'
            f'<div class="stat-row" style="margin:0">'
            f'<div class="stat"><span class="stat-label">Option A ade</span>'
            f'<span class="stat-val">{_fmt(alt["ade_a"])}<i>m</i></span></div>'
            f'<div class="stat"><span class="stat-label">Option B ade</span>{ade_b_html}</div>'
            f'</div>{degenerate_flag}'
            f'<ul class="quotes" style="margin-top:0.3rem">{cot_html}</ul>'
            f'</div>'
        )

    def position_block(pos: dict) -> str:
        alts_html = "".join(alt_row(a, pos["sampled_prob"]) for a in pos["alternatives"])
        return (
            f'<details class="scene"><summary>'
            f'<span class="scene-t0">step {pos["step"]}</span>'
            f'<span class="scene-n">sampled &lsquo;{_esc(pos["sampled_token"].strip())}&rsquo; '
            f'(p={_fmt(pos["sampled_prob"], 2)}, H={_fmt(pos["entropy"], 2)})</span>'
            f'</summary><div class="scene-body">{alts_html}</div></details>'
        )

    def scene_block(scene: dict) -> str:
        positions_html = "".join(position_block(p) for p in scene["positions"])
        return (
            f'<details class="clip" data-clipid="{_esc(scene["scene_id"])}"><summary>'
            f'<span class="clip-id">{_esc(scene["scene_id"])}</span>'
            f'<span class="clip-scene-count">{scene["n_positions"]} positions '
            f'&middot; max clean ADE {_fmt(scene["max_clean_ade"])} m</span></summary>'
            f'<div class="clip-body"><p class="masthead-meta" style="margin:0.5rem 0">'
            f'<q>{_esc(scene["cot"])}</q></p>{positions_html}</div></details>'
        )

    scenes_html = "".join(scene_block(s) for s in data["scenes"])

    return (
        f'<div class="masthead-stats" style="margin:0 0 1.5rem">{stat_tiles}</div>'
        f'{compare_html}'
        f'<section class="cluster-section"><details open><summary>'
        f'<h2>Per-scene drill-down</h2>'
        f'<span class="cluster-count">sorted by max clean trajectory delta</span></summary>'
        f'<div class="cluster-body">{scenes_html}</div></details></section>'
    )
