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


def example_key(scene_id: str, step: int, token: str) -> str:
    """Filename-safe key shared with render_examples.py (which writes
    {key}.png) so a rendered trajectory-comparison plot can be matched back
    to its alternative row. Lives here (zero dependencies) rather than in
    render_examples.py so this HTML-string-building module never needs to
    import matplotlib just to compute a filename key."""
    safe_token = "".join(c if c.isalnum() else "_" for c in token.strip()) or "blank"
    return f"{scene_id}_step{step}_{safe_token}"


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
    clean_a_step0: list[float] = []
    clean_a_other: list[float] = []
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
                (clean_a_step0 if step == 0 else clean_a_other).append(alt_a["traj_ade_m"])
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
        "stats_a_step0": _stats(clean_a_step0),
        "stats_a_other": _stats(clean_a_other),
        "stats_b": _stats(clean_b),
        "stats_b_step0": _stats(clean_b_step0),
        "stats_b_other": _stats(clean_b_other),
    }


def render_counterfactual_section(data: dict[str, Any], example_plots_b64: dict[str, str] | None = None) -> str:
    """Render the full 'Token Sensitivity' tab body (everything inside the
    <div data-tab-panel="...">, not the page shell).

    example_plots_b64: optional {example_key: base64 PNG} for the small
    curated set of (scene_id, step, token) examples that have a rendered
    top-down trajectory comparison plot (see render_examples.py). Alternative
    rows without a matching entry render exactly as before -- this is a
    deliberately sparse, opt-in overlay onto the full per-scene drill-down,
    not something every one of the ~6944 rows is expected to have."""
    example_plots_b64 = example_plots_b64 or {}

    sa, sb = data["stats_a"], data["stats_b"]
    sa0, sao = data["stats_a_step0"], data["stats_a_other"]
    sb0, sbo = data["stats_b_step0"], data["stats_b_other"]

    stat_tiles = "".join(
        f'<div class="masthead-stat"><b>{_esc(v)}</b><span>{_esc(label)}</span></div>'
        for v, label in [
            (data["n_scenes"], "scenes"), (data["n_positions"], "positions swept"),
            (data["n_alternatives"], "alternatives tested"), (data["n_degenerate"], "degenerate (excluded)"),
        ]
    )

    compare_html = (
        f'<div class="compare-legend">'
        f'<span class="legend-chip"><i class="swatch swatch-a"></i>Option A (isolated single-token swap)</span>'
        f'<span class="legend-chip"><i class="swatch swatch-b"></i>Option B (forced token, reasoning re-sampled)</span>'
        f'</div>'
        f'<p class="compare-missing-note">'
        f'Full distributions (m, ADE, clean/non-degenerate only), mean / median / p90 / max &mdash; '
        f'Option A overall: {_fmt(sa["mean"])} / {_fmt(sa["median"])} / {_fmt(sa["p90"])} / {_fmt(sa["max"])}, '
        f'step 0: {_fmt(sa0["mean"])} / {_fmt(sa0["median"])} / {_fmt(sa0["p90"])} / {_fmt(sa0["max"])}, '
        f'later steps: {_fmt(sao["mean"])} / {_fmt(sao["median"])} / {_fmt(sao["p90"])} / {_fmt(sao["max"])}. '
        f'Option B overall: {_fmt(sb["mean"])} / {_fmt(sb["median"])} / {_fmt(sb["p90"])} / {_fmt(sb["max"])}, '
        f'step 0: {_fmt(sb0["mean"])} / {_fmt(sb0["median"])} / {_fmt(sb0["p90"])} / {_fmt(sb0["max"])}, '
        f'later steps: {_fmt(sbo["mean"])} / {_fmt(sbo["median"])} / {_fmt(sbo["p90"])} / {_fmt(sbo["max"])}. '
        f'{data["n_degenerate"]} of {data["n_alternatives"]} Option B continuations never closed their reasoning '
        f'cleanly within the generation budget and are excluded from every statistic above (mean ADE among '
        f'those degenerate ones alone: ~25&times; higher than clean ones &mdash; a KV-cache/diffusion artifact '
        f'of incomplete generation, not a reasoning-sensitivity signal).'
        f'</p>'
    )

    def alt_row(scene_id: str, step: int, alt: dict) -> tuple[str, bool]:
        degenerate_flag = '<span class="incomplete-flag">generation incomplete</span>' if alt["degenerate"] else ""
        ade_b_html = (
            f'<span class="stat-val">{_fmt(alt["ade_b"])}<i>m</i></span>' if alt["ade_b"] is not None and not alt["degenerate"]
            else '<span class="stat-val" style="color:var(--ink-dim)">&mdash;</span>'
        )
        cot_html = ""
        if alt["forced_cot"] and not alt["degenerate"]:
            cot_html = f'<li><q>{_esc(alt["forced_cot"])}</q></li>'
        plot_b64 = example_plots_b64.get(example_key(scene_id, step, alt["token"]))
        has_plot = plot_b64 is not None
        plot_html = (
            f'<details class="cf-plot-toggle" open><summary>Hide trajectory plot</summary>'
            f'<img class="scene-traj-img" loading="lazy" '
            f'alt="Top-down baseline vs. counterfactual trajectory comparison" '
            f'src="data:image/png;base64,{plot_b64}"></details>'
            if has_plot else ""
        )
        badge = ' <span class="cf-has-plot-badge">&#128200; trajectory plot below</span>' if has_plot else ""
        row_html = (
            f'<div class="cf-alt-row">'
            f'<span class="cf-alt-token">&rarr; &lsquo;{_esc(alt["token"].strip())}&rsquo;</span>'
            f'<span class="cf-alt-prob">p={_fmt(alt["prob"], 2)}</span>'
            f'{badge}'
            f'<div class="stat-row" style="margin:0">'
            f'<div class="stat"><span class="stat-label">Option A ade</span>'
            f'<span class="stat-val">{_fmt(alt["ade_a"])}<i>m</i></span></div>'
            f'<div class="stat"><span class="stat-label">Option B ade</span>{ade_b_html}</div>'
            f'</div>{degenerate_flag}'
            f'<ul class="quotes" style="margin-top:0.3rem">{cot_html}</ul>'
            f'{plot_html}'
            f'</div>'
        )
        return row_html, has_plot

    def position_block(scene_id: str, pos: dict) -> tuple[str, bool]:
        rows = [alt_row(scene_id, pos["step"], a) for a in pos["alternatives"]]
        alts_html = "".join(r[0] for r in rows)
        has_plot = any(r[1] for r in rows)
        badge = ' <span class="cf-has-plot-badge">&#128200; has plot</span>' if has_plot else ""
        open_attr = " open" if has_plot else ""
        return (
            f'<details class="scene"{open_attr}><summary>'
            f'<span class="scene-t0">step {pos["step"]}</span>'
            f'<span class="scene-n">sampled &lsquo;{_esc(pos["sampled_token"].strip())}&rsquo; '
            f'(p={_fmt(pos["sampled_prob"], 2)}, H={_fmt(pos["entropy"], 2)})</span>{badge}'
            f'</summary><div class="scene-body">{alts_html}</div></details>',
            has_plot,
        )

    def scene_block(scene: dict) -> str:
        blocks = [position_block(scene["scene_id"], p) for p in scene["positions"]]
        positions_html = "".join(b[0] for b in blocks)
        has_plot = any(b[1] for b in blocks)
        badge = ' <span class="cf-has-plot-badge">&#128200; has trajectory plot(s)</span>' if has_plot else ""
        open_attr = " open" if has_plot else ""
        return (
            f'<details class="clip" data-clipid="{_esc(scene["scene_id"])}"{open_attr}><summary>'
            f'<span class="clip-id">{_esc(scene["scene_id"])}</span>'
            f'<span class="clip-scene-count">{scene["n_positions"]} positions '
            f'&middot; max clean ADE {_fmt(scene["max_clean_ade"])} m</span>{badge}</summary>'
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
