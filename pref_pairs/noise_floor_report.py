# SPDX-License-Identifier: Apache-2.0
"""
noise_floor_report.py — consolidates action_space_variance_report.json
(per-scene stats) and scene_reasoning/*_reasoning.md (per-scene rollout
reasoning, from scene_reasoning_report.py) into ONE self-contained HTML page:
category -> clip -> scene, with the calibrated noise floor (epsilon) shown
at a glance per category and representative reasoning excerpts per scene.

Why this exists: those two artifacts already have everything a reviewer
needs, but spread across one JSON file and ~120 separate per-scene Markdown
files -- there's no single view of "for this scenario category, what's the
noise floor, and for this specific clip, what did the model actually say
across its repeated rollouts." This module computes nothing new; it's purely
a reformatting/aggregation layer over data both upstream reports already
produced.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# (json column in per_clip -> (short key, display label, unit)) for the four
# metrics action_space_variance.py's report already recommends epsilon from.
METRICS: dict[str, tuple[str, str, str]] = {
    "accel_std_mean_over_waypoints": ("accel_std", "accel σ", "m/s²"),
    "curvature_std_mean_over_waypoints": ("curvature_std", "curvature σ", "1/m"),
    "final_lateral_offset_m_std": ("lat_offset_std", "lateral offset σ", "m"),
    "total_heading_change_deg_std": ("heading_std", "heading Δ σ", "deg"),
}

_CLASS_COUNTS_RE = re.compile(r"^Class counts: (.+)$", re.M)
_CLASS_SECTION_RE = re.compile(r"^## (.+?) \(\d+ rollouts\)$", re.M)
_QUOTE_RE = re.compile(r"^> (.+)$", re.M)


def parse_reasoning_md(path: Path) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """Extract a scene's maneuver-class distribution and, per class, its
    reasoning quotes, from a scene_reasoning_report.py-produced Markdown
    file. Each class's rollouts rarely phrase their reasoning byte-identically
    twice, so quotes_by_class[cls]["top"] is only the 3 LARGEST exact-text
    repeat clusters, not a partition of all rollouts in that class -- e.g. a
    class of 100 rollouts commonly has 60-80 distinct phrasings, most said
    only once. n_unique/n_rollouts are included so callers can render that
    context instead of leaving readers to assume the shown counts sum to the
    class total (they don't, by design)."""
    text = path.read_text()
    class_counts: dict[str, int] = {}
    m = _CLASS_COUNTS_RE.search(text)
    if m:
        for part in m.group(1).split(", "):
            cls, n = part.split("=")
            class_counts[cls] = int(n)

    sections = _CLASS_SECTION_RE.split(text)[1:]  # [class, body, class, body, ...]
    quotes_by_class: dict[str, dict[str, Any]] = {}
    for cls, body in zip(sections[0::2], sections[1::2]):
        counts = collections.Counter(_QUOTE_RE.findall(body))
        quotes_by_class[cls] = {
            "top": counts.most_common(3),
            "n_unique": len(counts),
            "n_rollouts": sum(counts.values()),
        }
    return class_counts, quotes_by_class


def _percentile(sorted_vals: list[float], q: float) -> float:
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _median(sorted_vals: list[float]) -> float:
    n = len(sorted_vals)
    mid = n // 2
    return sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


def build_report_data(results_dir: str | Path) -> dict[str, Any]:
    """Load action_space_variance_report.json + scene_reasoning/*.md from
    results_dir and return the nested category -> clip -> scene structure
    the HTML template renders. Raises if per_clip has duplicate scene_ids --
    that's fetch_from_logs.py's dedup bug, not something to silently paper
    over here."""
    results_dir = Path(results_dir)
    with open(results_dir / "action_space_variance_report.json") as f:
        report = json.load(f)
    per_clip = report["per_clip"]

    scene_ids = [r["scene_id"] for r in per_clip]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError(
            "per_clip has duplicate scene_ids -- re-run fetch_from_logs.py "
            "(its drop_duplicates fix) before building this report."
        )

    reasoning_dir = results_dir / "scene_reasoning"
    clusters: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in per_clip:
        clusters[row["event_cluster"]][row["clip_id"]].append(row)

    data: dict[str, Any] = {"clusters": {}}
    for cluster, clips in clusters.items():
        all_rows = [r for rows in clips.values() for r in rows]
        cluster_stats = {}
        for col, (short, _, _) in METRICS.items():
            vals = sorted(r[col] for r in all_rows)
            cluster_stats[short] = {
                "median": _median(vals), "p90": _percentile(vals, 0.9),
                "min": vals[0], "max": vals[-1],
            }

        clip_entries = []
        for clip_id, rows in sorted(clips.items()):
            scene_entries = []
            for row in sorted(rows, key=lambda r: r["t0_us"]):
                md_path = reasoning_dir / f"{row['scene_id']}_reasoning.md"
                reasoning = None
                if md_path.exists():
                    class_counts, quotes_by_class = parse_reasoning_md(md_path)
                    reasoning = {"class_counts": class_counts, "quotes_by_class": quotes_by_class}
                scene_entries.append({
                    "scene_id": row["scene_id"],
                    "t0_us": row["t0_us"],
                    "n_rollouts": row["n_rollouts"],
                    "complete": row["complete"],
                    "stats": {short: row[col] for col, (short, _, _) in METRICS.items()},
                    "reasoning": reasoning,
                })
            clip_entries.append({"clip_id": clip_id, "scenes": scene_entries})

        data["clusters"][cluster] = {
            "n_clips": len(clip_entries),
            "n_scenes": len(all_rows),
            "stats": cluster_stats,
            "clips": clip_entries,
        }
    return data


def _esc(v: Any) -> str:
    return html.escape(str(v))


def _fmt(v: float, sig: int = 3) -> str:
    return "0" if v == 0 else f"{v:.{sig}g}"


def _cluster_label(cluster: str) -> str:
    return cluster.replace("_", " ").title().replace("Or", "or")


def render_section(
    data: dict[str, Any],
    video_b64_by_scene: dict[str, str] | None = None,
    image_b64_by_scene: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Render one dataset's gauge cards + category/clip/scene drill-down as
    HTML fragments (not a full page). Split out of render_html so a second
    dataset (e.g. a fixed-reasoning/diffusion-only run's results, see
    pref_pairs/fixed_reasoning_rollout.py) can be rendered as a second tab
    with zero changes to any of the per-cluster/per-clip/per-scene logic
    below -- only render_html's page-assembly needs to know there are two.

    video_b64_by_scene: optional {scene_id: base64-encoded mp4 bytes} --
    scenes with an entry get an inline <video> (data: URI, so the page stays
    self-contained per the Artifact CSP); scenes without one show no video.
    Deliberately NOT auto-discovered from a directory here -- rendering a
    video is expensive (S3 + log fetches per scene, see
    render_trajectory_overlay.py) and this keeps that cost an explicit
    caller decision, not something that silently scales with report size.

    image_b64_by_scene: optional {scene_id: base64-encoded PNG bytes} -- the
    top-down, colored-by-maneuver-class trajectory plot scene_reasoning_report.py
    already produces for every scene (all rollouts overlaid). Unlike video, this
    is UNIQUE PER DATASET (dataset A and B have different scene_ids' plots even
    though the underlying clip/manifest is shared), so it is NOT the same shared
    dict passed to both render_section calls the way video_b64_by_scene is.

    Returns {"gauge_cards", "cluster_sections", "total_clips", "total_scenes", "n_categories"}.
    """
    video_b64_by_scene = video_b64_by_scene or {}
    image_b64_by_scene = image_b64_by_scene or {}
    cluster_order = sorted(data["clusters"].items(), key=lambda kv: -kv[1]["n_scenes"])
    total_clips = sum(v["n_clips"] for _, v in cluster_order)
    total_scenes = sum(v["n_scenes"] for _, v in cluster_order)

    def gauge_card(cluster: str, v: dict) -> str:
        n = v["n_clips"]
        warn_attr = ' data-lown="1"' if n < 5 else ""
        rows = ""
        for _, (short, label, unit) in METRICS.items():
            s = v["stats"][short]
            med, p90, mn, mx = s["median"], s["p90"], s["min"], s["max"]

            def pct_of(x: float) -> float:
                return 100.0 if mx <= 0 else max(0.0, min(100.0, (x / mx) * 100))

            min_pct, med_pct, p90_pct = pct_of(mn), pct_of(med), pct_of(p90)
            rows += (
                f'<div class="gauge-row">'
                f'<div class="gauge-row-main"><span class="gauge-label">{_esc(label)}</span>'
                f'<div class="gauge-track">'
                f'<div class="gauge-range" style="left:{min_pct:.0f}%"></div>'
                f'<div class="gauge-median-tick" style="left:{med_pct:.0f}%"></div>'
                f'<div class="gauge-p90-tick" style="left:{p90_pct:.0f}%"></div>'
                f'</div>'
                f'<span class="gauge-vals"><b>{_fmt(p90)}</b><i>{_esc(unit)}</i></span></div>'
                f'<div class="gauge-row-detail">'
                f'<span>min {_fmt(mn)}</span><span>med {_fmt(med)}</span>'
                f'<span class="gauge-p90-label">p90 {_fmt(p90)}</span><span>max {_fmt(mx)}</span>'
                f'</div></div>'
            )
        low_n_note = " <em>low-n</em>" if n < 5 else ""
        return (
            f'<article class="gauge-card"{warn_attr}><header><h3>{_esc(_cluster_label(cluster))}</h3>'
            f'<span class="gauge-count">{n} clip{"s" if n != 1 else ""} · {v["n_scenes"]} scene'
            f'{"s" if v["n_scenes"] != 1 else ""}{low_n_note}</span></header>{rows}</article>'
        )

    def reasoning_block(reasoning: dict | None) -> str:
        if reasoning is None:
            return '<p class="no-reasoning">No detailed rollout log captured for this scene (outside this run\'s per-rank detail cap).</p>'
        cc = reasoning["class_counts"]
        total = sum(cc.values()) or 1
        chips = "".join(
            f'<span class="class-chip" style="--w:{100 * n / total:.0f}%"><b>{n}</b> {_esc(cls)}</span>'
            for cls, n in sorted(cc.items(), key=lambda kv: -kv[1])
        )
        quotes_html = ""
        for cls, qinfo in reasoning["quotes_by_class"].items():
            top = qinfo["top"]
            if not top:
                continue
            q_html = ""
            for q, n in top:
                count_span = f'<span class="qcount">×{n}</span>' if n > 1 else ""
                q_html += f"<li><q>{_esc(q)}</q> {count_span}</li>"
            coverage_note = (
                f'<span class="quote-coverage">top {len(top)} of {qinfo["n_unique"]} distinct phrasings '
                f'among {qinfo["n_rollouts"]} rollouts &mdash; counts below don\'t sum to {qinfo["n_rollouts"]}, '
                f'most rollouts phrase it slightly differently each time</span>'
            )
            quotes_html += (
                f'<div class="quote-group"><span class="quote-class-label">{_esc(cls)}</span>'
                f'{coverage_note}<ul>{q_html}</ul></div>'
            )
        flag = '<span class="divergent-flag">rollouts diverged into multiple maneuvers</span>' if len(cc) > 1 else ""
        return f'<div class="class-counts">{chips}{flag}</div><div class="quotes">{quotes_html}</div>'

    def scene_block(scene: dict) -> str:
        s = scene["stats"]
        stat_html = "".join(
            f'<div class="stat"><span class="stat-label">{_esc(label)}</span>'
            f'<span class="stat-val">{_fmt(s[short])}<i>{_esc(unit)}</i></span></div>'
            for _, (short, label, unit) in METRICS.items()
        )
        incomplete = "" if scene["complete"] else '<span class="incomplete-flag">incomplete rollout set</span>'
        image_b64 = image_b64_by_scene.get(scene["scene_id"])
        image_html = (
            f'<img class="scene-traj-img" loading="lazy" '
            f'alt="Top-down trajectories for all rollouts in this scene, colored by maneuver class" '
            f'src="data:image/png;base64,{image_b64}">'
            if image_b64 else ""
        )
        video_b64 = video_b64_by_scene.get(scene["scene_id"])
        video_html = (
            f'<video class="scene-video" controls preload="none" playsinline muted loop>'
            f'<source src="data:video/mp4;base64,{video_b64}" type="video/mp4"></video>'
            if video_b64 else ""
        )
        return (
            f'<details class="scene"><summary><span class="scene-t0">t0={_esc(scene["t0_us"])}µs</span>'
            f'<span class="scene-n">{_esc(scene["n_rollouts"])} rollouts</span>{incomplete}</summary>'
            f'<div class="scene-body">{image_html}{video_html}<div class="stat-row">{stat_html}</div>'
            f'{reasoning_block(scene["reasoning"])}</div></details>'
        )

    def clip_block(clip: dict) -> str:
        scenes_html = "".join(scene_block(s) for s in clip["scenes"])
        n = len(clip["scenes"])
        return (
            f'<details class="clip" data-clipid="{_esc(clip["clip_id"])}"><summary>'
            f'<span class="clip-id">{_esc(clip["clip_id"])}</span>'
            f'<span class="clip-scene-count">{n} scene{"s" if n != 1 else ""}</span></summary>'
            f'<div class="clip-body">{scenes_html}</div></details>'
        )

    def cluster_section(cluster: str, v: dict, open_attr: str) -> str:
        clips_html = "".join(clip_block(c) for c in v["clips"])
        return (
            f'<section class="cluster-section" id="cluster-{_esc(cluster)}"><details {open_attr}>'
            f'<summary><h2>{_esc(_cluster_label(cluster))}</h2>'
            f'<span class="cluster-count">{v["n_clips"]} clips · {v["n_scenes"]} scenes</span></summary>'
            f'<div class="cluster-body">{clips_html}</div></details></section>'
        )

    gauge_cards = "".join(gauge_card(c, v) for c, v in cluster_order)
    cluster_sections = "".join(
        cluster_section(c, v, "open" if i == 0 else "") for i, (c, v) in enumerate(cluster_order)
    )

    return {
        "gauge_cards": gauge_cards,
        "cluster_sections": cluster_sections,
        "total_clips": total_clips,
        "total_scenes": total_scenes,
        "n_categories": len(cluster_order),
    }


def render_comparison_section(data: dict[str, Any], data_b: dict[str, Any], label: str, label_b: str) -> str:
    """Render the 'Compare' tab: one small-multiple panel per calibration
    metric (never a shared/dual axis across metrics with different units --
    that's the #1 dataviz anti-pattern), each panel a dumbbell row per
    scenario category plotting dataset A's and B's recommended epsilon (p90
    across scenes) on ONE shared x-scale for that metric. This is what lets a
    reader see at a glance, per category, whether fixing reasoning shrinks or
    grows the noise floor and by how much -- flipping between two separate
    tabs and remembering numbers does not give that.

    p90 (not median) is compared here because it's already the number the
    gauge cards headline as "the recommended epsilon" -- this isn't a new
    statistic, just the same one plotted for both datasets at once.

    Colors (--cmp-a/--cmp-b) are a categorical pair validated separately from
    --accent/--teal (see the page's :root block) since those two already mean
    something else (p90/median ticks WITHIN one gauge) -- reusing them here
    for dataset identity would overload the same hues with two meanings on
    one page.
    """
    clusters_a, clusters_b = data["clusters"], data_b["clusters"]
    shared = sorted(set(clusters_a) & set(clusters_b))
    missing = sorted((set(clusters_a) | set(clusters_b)) - set(shared))
    if missing:
        logger.warning("compare view: category present in only one dataset, omitted: %s", missing)

    panels = ""
    for col, (short, mlabel, unit) in METRICS.items():
        rows = [
            (cluster, clusters_a[cluster]["stats"][short]["p90"], clusters_b[cluster]["stats"][short]["p90"])
            for cluster in shared
        ]
        rows.sort(key=lambda r: -max(r[1], r[2]))
        scale_max = max((max(pa, pb) for _, pa, pb in rows), default=1.0) or 1.0

        def pct(x: float) -> float:
            return max(0.0, min(100.0, 100.0 * x / scale_max))

        row_html = ""
        for cluster, pa, pb in rows:
            xa, xb = pct(pa), pct(pb)
            lo, hi = min(xa, xb), max(xa, xb)
            row_html += (
                f'<div class="dumbbell-row"><span class="dumbbell-label">{_esc(_cluster_label(cluster))}</span>'
                f'<div class="dumbbell-track">'
                f'<div class="dumbbell-line" style="left:{lo:.1f}%;width:{(hi - lo):.1f}%"></div>'
                f'<div class="dumbbell-dot dot-a" style="left:{xa:.1f}%" '
                f'title="{_esc(label)}: {_fmt(pa)} {_esc(unit)}"></div>'
                f'<div class="dumbbell-dot dot-b" style="left:{xb:.1f}%" '
                f'title="{_esc(label_b)}: {_fmt(pb)} {_esc(unit)}"></div>'
                f'</div>'
                f'<span class="dumbbell-vals"><i class="swatch swatch-a"></i>{_fmt(pa)}'
                f'<i class="swatch swatch-b"></i>{_fmt(pb)}<i class="dumbbell-unit">{_esc(unit)}</i></span>'
                f'</div>'
            )
        panels += (
            f'<article class="compare-panel"><h3>{_esc(mlabel)}'
            f'<span class="compare-metric-note">p90 epsilon &middot; {_esc(unit)}</span></h3>{row_html}</article>'
        )

    missing_note = ""
    if missing:
        names = ", ".join(_cluster_label(c) for c in missing)
        missing_note = (
            f'<p class="compare-missing-note">{len(missing)} categor{"y" if len(missing) == 1 else "ies"} '
            f"present in only one dataset, omitted from this view: {_esc(names)}</p>"
        )

    return (
        f'<div class="compare-legend">'
        f'<span class="legend-chip"><i class="swatch swatch-a"></i>{_esc(label)}</span>'
        f'<span class="legend-chip"><i class="swatch swatch-b"></i>{_esc(label_b)}</span>'
        f"</div>"
        f'<div class="compare-grid">{panels}</div>{missing_note}'
    )


def render_html(
    data: dict[str, Any],
    label: str = "Compound noise (reasoning + diffusion)",
    *,
    data_b: dict[str, Any] | None = None,
    label_b: str = "Diffusion-only noise (reasoning fixed)",
    video_b64_by_scene: dict[str, str] | None = None,
    image_b64_by_scene_a: dict[str, str] | None = None,
    image_b64_by_scene_b: dict[str, str] | None = None,
    counterfactual_html: str | None = None,
    counterfactual_label: str = "Token Sensitivity",
) -> str:
    """Render one or two datasets as a single self-contained HTML page.

    With just `data`: identical single-tab page as before this function
    supported a second dataset -- no tab bar, no behavior change for
    existing callers (e.g. the published Task 3 report).

    With `data_b` also given: a tab bar switches between three panels -- the
    two full panels (each its own gauge grid + category/clip/scene drill-down),
    e.g. Task 3's compound-noise numbers vs. the fixed-reasoning mode's
    diffusion-only numbers on the same scenes, PLUS a third "Compare" panel
    that plots both side by side (see render_comparison_section). The clip-ID
    filter is shared (one input) but scoped to whichever of the two drill-down
    panels is currently visible, and hidden entirely on the Compare panel
    (there's nothing there to filter by clip ID).

    counterfactual_html: an optional pre-rendered HTML fragment (see
    counterfactual.report.render_counterfactual_section) shown as a 4th tab.
    Independent of data_b/has_tabs -- this is a different experiment
    (per-token logit counterfactuals) on the same manifest, not another
    noise-floor dataset, so it's passed in fully rendered rather than going
    through render_section. Its scene-level <details class="clip"> elements
    reuse the existing clip-ID filter/JS as-is (no filter-hiding needed here,
    unlike the Compare tab, since these ARE filterable by scene/clip id).
    """
    section_a = render_section(data, video_b64_by_scene, image_b64_by_scene_a)
    has_tabs = data_b is not None
    section_b = render_section(data_b, video_b64_by_scene, image_b64_by_scene_b) if has_tabs else None

    tab_bar_buttons = []
    compare_html = ""
    if has_tabs:
        tab_bar_buttons = [
            f'<button type="button" class="tab-btn" data-tab-btn="a" aria-selected="true">{_esc(label)}</button>',
            f'<button type="button" class="tab-btn" data-tab-btn="b" aria-selected="false">{_esc(label_b)}</button>',
            '<button type="button" class="tab-btn" data-tab-btn="compare" aria-selected="false">Compare</button>',
        ]
        compare_html = render_comparison_section(data, data_b, label, label_b)
    if counterfactual_html is not None:
        first = not tab_bar_buttons
        tab_bar_buttons.append(
            f'<button type="button" class="tab-btn" data-tab-btn="counterfactual" '
            f'aria-selected="{"true" if first else "false"}">{_esc(counterfactual_label)}</button>'
        )
    tab_bar = f'<div class="tab-bar" role="tablist">{"".join(tab_bar_buttons)}</div>' if tab_bar_buttons else ""

    def panel_html(tab: str, section: dict[str, Any], hidden: bool) -> str:
        hidden_attr = " hidden" if hidden else ""
        return (
            f'<div data-tab-panel="{tab}"{hidden_attr}>'
            f'<section aria-label="Per-category noise floor"><div class="gauge-grid">{section["gauge_cards"]}</div></section>'
            f'<main>{section["cluster_sections"]}</main>'
            "</div>"
        )

    # "a" is the default visible tab whenever it has its own tab button
    # (has_tabs=True) or there's no tab bar at all (single-dataset caller).
    # If counterfactual_html is given WITHOUT data_b, the counterfactual tab
    # becomes the sole/first button instead, so "a" starts hidden then.
    a_is_default = has_tabs or counterfactual_html is None
    panels = panel_html("a", section_a, hidden=not a_is_default)
    if has_tabs:
        panels += panel_html("b", section_b, hidden=True)
        panels += f'<div data-tab-panel="compare" hidden>{compare_html}</div>'
    if counterfactual_html is not None:
        panels += f'<div data-tab-panel="counterfactual"{"" if not a_is_default else " hidden"}>{counterfactual_html}</div>'

    return _PAGE_TEMPLATE.format(
        total_clips=section_a["total_clips"],
        total_scenes=section_a["total_scenes"],
        n_categories=section_a["n_categories"],
        tab_bar=tab_bar,
        panels=panels,
    )


_PAGE_TEMPLATE = """<title>Action-Space Noise Floor Report</title>
<style>
:root {{
  --paper: #E8E9E4; --paper-raised: #F4F4F0; --ink: #20242B; --ink-dim: #565C63;
  --line: #C9CBC3; --accent: #B96A22; --accent-ink: #FFFFFF; --teal: #2E7671;
  --warn: #A63F35; --good: #4C6A45;
  /* Two-series categorical pair for the A-vs-B compare chart, distinct from
     --accent/--teal (those already carry a different meaning -- p90/median
     ticks WITHIN one gauge). Validated via dataviz skill's validate_palette.js
     against this surface (--paper-raised) in both modes: lightness band,
     chroma floor, CVD separation (worst adjacent dE ~100), contrast all pass. */
  --cmp-a: #2a78d6; --cmp-b: #eb6834;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
  --sans: ui-sans-serif, "Neue Haas Grotesk Text", "Helvetica Neue", Arial, sans-serif;
  --serif: Georgia, "Iowan Old Style", "Times New Roman", serif;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --paper: #14171B; --paper-raised: #1B1F24; --ink: #E7E5DD; --ink-dim: #9B9F97;
    --line: #363C42; --accent: #E0954A; --accent-ink: #14171B; --teal: #55A69F;
    --warn: #D9776A; --good: #85A377; --cmp-a: #3987e5; --cmp-b: #d95926; }}
}}
:root[data-theme="dark"] {{ --paper: #14171B; --paper-raised: #1B1F24; --ink: #E7E5DD; --ink-dim: #9B9F97;
  --line: #363C42; --accent: #E0954A; --accent-ink: #14171B; --teal: #55A69F; --warn: #D9776A; --good: #85A377;
  --cmp-a: #3987e5; --cmp-b: #d95926; }}
:root[data-theme="light"] {{ --paper: #E8E9E4; --paper-raised: #F4F4F0; --ink: #20242B; --ink-dim: #565C63;
  --line: #C9CBC3; --accent: #B96A22; --accent-ink: #FFFFFF; --teal: #2E7671; --warn: #A63F35; --good: #4C6A45;
  --cmp-a: #2a78d6; --cmp-b: #eb6834; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: var(--sans); line-height: 1.45; padding: 0 0 6rem; }}
::selection {{ background: var(--accent); color: var(--accent-ink); }}
a {{ color: var(--teal); }}
.page {{ max-width: 74rem; margin: 0 auto; padding: 0 1.5rem; }}
header.masthead {{ padding: 2.75rem 0 2rem; border-bottom: 1px solid var(--line); display: flex; flex-direction: column; gap: 0.6rem; }}
.eyebrow {{ font: 700 0.72rem/1 var(--sans); letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font: 600 2.15rem/1.15 var(--sans); margin: 0; letter-spacing: -0.01em; text-wrap: balance; }}
.masthead-meta {{ font: 0.92rem/1.5 var(--sans); color: var(--ink-dim); max-width: 62ch; }}
.masthead-meta code {{ font-family: var(--mono); background: var(--paper-raised); padding: 0.05rem 0.35rem; border-radius: 3px; }}
.masthead-stats {{ display: flex; gap: 1.75rem; margin-top: 0.4rem; }}
.masthead-stat b {{ font: 600 1.5rem/1 var(--sans); font-variant-numeric: tabular-nums; display: block; }}
.masthead-stat span {{ font-size: 0.78rem; color: var(--ink-dim); text-transform: uppercase; letter-spacing: 0.06em; }}
.gauge-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr)); gap: 1rem; margin: 2rem 0 2.5rem; }}
.gauge-card {{ background: var(--paper-raised); border: 1px solid var(--line); border-radius: 10px; padding: 1.1rem 1.2rem 1.3rem; }}
.gauge-card[data-lown="1"] {{ border-color: color-mix(in srgb, var(--warn) 45%, var(--line)); }}
.gauge-card header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 0.75rem; margin-bottom: 0.85rem; }}
.gauge-card h3 {{ font: 600 1rem/1.25 var(--sans); margin: 0; text-wrap: balance; }}
.gauge-count {{ font: 0.72rem var(--mono); color: var(--ink-dim); white-space: nowrap; font-variant-numeric: tabular-nums; }}
.gauge-count em {{ color: var(--warn); font-style: normal; font-weight: 600; }}
.gauge-row {{ margin: 0.55rem 0; }}
.gauge-row-main {{ display: grid; grid-template-columns: 6.4rem 1fr auto; align-items: center; gap: 0.6rem; }}
.gauge-label {{ font-size: 0.74rem; color: var(--ink-dim); }}
.gauge-track {{ position: relative; height: 6px; background: var(--line); border-radius: 4px; overflow: visible; }}
.gauge-range {{ position: absolute; top: 0; bottom: 0; right: 0; background: color-mix(in srgb, var(--teal) 40%, transparent); border-radius: 4px; }}
.gauge-median-tick {{ position: absolute; top: -3px; width: 2px; height: 12px; background: var(--teal); border-radius: 1px; transform: translateX(-1px); }}
.gauge-p90-tick {{ position: absolute; top: -3px; width: 2px; height: 12px; background: var(--accent); border-radius: 1px; transform: translateX(-1px); }}
.gauge-vals {{ font: 0.78rem var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }}
.gauge-vals i {{ font-style: normal; color: var(--ink-dim); margin-right: 0.35rem; }}
.gauge-row-detail {{ display: flex; gap: 0.7rem; margin: 0.25rem 0 0 6.4rem; font: 0.68rem var(--mono); color: var(--ink-dim); font-variant-numeric: tabular-nums; }}
.gauge-p90-label {{ color: var(--accent); }}
.tab-bar {{ display: flex; gap: 0.4rem; margin: 1.75rem 0 0; border-bottom: 1px solid var(--line); }}
.tab-btn {{ font: 600 0.86rem var(--sans); color: var(--ink-dim); background: none; border: none; border-bottom: 2px solid transparent; padding: 0.7rem 0.2rem; margin-right: 1.4rem; cursor: pointer; }}
.tab-btn:hover {{ color: var(--ink); }}
.tab-btn[aria-selected="true"] {{ color: var(--ink); border-bottom-color: var(--accent); }}
.tab-btn:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.filter-bar {{ position: sticky; top: 0; z-index: 5; background: var(--paper); padding: 0.9rem 0; border-bottom: 1px solid var(--line); margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.75rem; }}
.filter-bar input {{ flex: 1; max-width: 26rem; font: 0.92rem var(--mono); padding: 0.55rem 0.8rem; background: var(--paper-raised); border: 1px solid var(--line); border-radius: 7px; color: var(--ink); }}
.filter-bar input:focus {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
.filter-count {{ font: 0.78rem var(--sans); color: var(--ink-dim); }}
.cluster-section > details > summary {{ list-style: none; cursor: pointer; display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; padding: 0.9rem 0; border-bottom: 1px solid var(--line); }}
.cluster-section > details > summary::-webkit-details-marker {{ display: none; }}
.cluster-section > details > summary::before {{ content: "▸"; color: var(--accent); margin-right: 0.6rem; font-size: 0.85em; display: inline-block; transition: transform 0.15s ease; }}
.cluster-section > details[open] > summary::before {{ transform: rotate(90deg); }}
.cluster-section h2 {{ font: 600 1.3rem var(--sans); margin: 0; display: inline; }}
.cluster-count {{ font: 0.8rem var(--mono); color: var(--ink-dim); white-space: nowrap; font-variant-numeric: tabular-nums; }}
.cluster-body {{ padding: 0.6rem 0 0.4rem 1.6rem; }}
details.clip {{ border-bottom: 1px dotted var(--line); }}
details.clip > summary {{ list-style: none; cursor: pointer; display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: 0.6rem 0; }}
details.clip > summary::-webkit-details-marker {{ display: none; }}
details.clip > summary::before {{ content: "＋"; color: var(--ink-dim); margin-right: 0.6rem; font-size: 0.75em; }}
details.clip[open] > summary::before {{ content: "－"; color: var(--accent); }}
.clip-id {{ font: 0.86rem var(--mono); }}
.clip-scene-count {{ font: 0.74rem var(--sans); color: var(--ink-dim); }}
.clip-body {{ padding: 0 0 0.75rem 1.4rem; display: flex; flex-direction: column; gap: 0.6rem; }}
details.scene {{ background: var(--paper-raised); border: 1px solid var(--line); border-radius: 8px; }}
details.scene > summary {{ list-style: none; cursor: pointer; display: flex; align-items: center; gap: 0.9rem; padding: 0.55rem 0.9rem; font-size: 0.82rem; }}
details.scene > summary::-webkit-details-marker {{ display: none; }}
.scene-t0 {{ font: 0.8rem var(--mono); font-variant-numeric: tabular-nums; }}
.scene-n {{ color: var(--ink-dim); font-size: 0.76rem; }}
.incomplete-flag, .divergent-flag {{ font: 600 0.66rem var(--sans); letter-spacing: 0.04em; text-transform: uppercase; color: var(--warn); border: 1px solid color-mix(in srgb, var(--warn) 55%, transparent); padding: 0.1rem 0.4rem; border-radius: 20px; }}
.scene-body {{ padding: 0 0.9rem 0.95rem; border-top: 1px solid var(--line); }}
.scene-video {{ display: block; width: 100%; max-width: 32rem; border-radius: 6px; margin-top: 0.75rem; background: #000; }}
.stat-row {{ display: flex; flex-wrap: wrap; gap: 1.1rem; margin: 0.75rem 0; }}
.stat {{ display: flex; flex-direction: column; gap: 0.1rem; }}
.stat-label {{ font-size: 0.68rem; color: var(--ink-dim); text-transform: uppercase; letter-spacing: 0.04em; }}
.stat-val {{ font: 600 0.95rem var(--mono); font-variant-numeric: tabular-nums; }}
.stat-val i {{ font: 0.68rem var(--sans); font-style: normal; color: var(--ink-dim); margin-left: 0.2rem; }}
.class-counts {{ display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; margin-bottom: 0.7rem; }}
.class-chip {{ font: 0.74rem var(--sans); background: var(--paper); border: 1px solid var(--line); border-radius: 20px; padding: 0.15rem 0.6rem; }}
.class-chip b {{ font-variant-numeric: tabular-nums; color: var(--teal); }}
.no-reasoning {{ font: 0.82rem var(--sans); color: var(--ink-dim); font-style: italic; margin: 0.5rem 0 0; }}
.quotes {{ display: flex; flex-direction: column; gap: 0.6rem; }}
.quote-group {{ display: flex; flex-direction: column; gap: 0.3rem; }}
.quote-class-label {{ font: 700 0.66rem var(--sans); letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-dim); }}
.quote-coverage {{ font: 0.72rem var(--sans); color: var(--ink-dim); display: block; margin: 0.1rem 0 0.35rem; }}
.quote-group ul {{ margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 0.3rem; }}
.quote-group li q {{ font: 0.92rem/1.5 var(--serif); font-style: italic; quotes: "“" "”"; }}
.qcount {{ font: 0.7rem var(--mono); color: var(--ink-dim); font-style: normal; }}
footer.page-footer {{ margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid var(--line); font: 0.78rem var(--sans); color: var(--ink-dim); }}
.scene-traj-img {{ display: block; width: 100%; max-width: 28rem; border-radius: 6px; margin-top: 0.75rem; background: var(--paper); border: 1px solid var(--line); }}
.swatch {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; flex: none; }}
.swatch-a {{ background: var(--cmp-a); }}
.swatch-b {{ background: var(--cmp-b); }}
.compare-legend {{ display: flex; gap: 1.4rem; margin: 1.75rem 0 0.25rem; }}
.legend-chip {{ display: flex; align-items: center; gap: 0.45rem; font: 0.84rem var(--sans); color: var(--ink-dim); }}
.compare-missing-note {{ font: 0.8rem var(--sans); color: var(--ink-dim); font-style: italic; margin: 0.6rem 0 0; }}
.compare-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(27rem, 1fr)); gap: 1.25rem; margin: 1.25rem 0 2rem; }}
.compare-panel {{ background: var(--paper-raised); border: 1px solid var(--line); border-radius: 10px; padding: 1.1rem 1.2rem 0.7rem; }}
.compare-panel h3 {{ font: 600 0.95rem var(--sans); margin: 0 0 0.9rem; }}
.compare-metric-note {{ font: 0.7rem var(--mono); color: var(--ink-dim); font-weight: 400; margin-left: 0.4rem; }}
.dumbbell-row {{ display: grid; grid-template-columns: 9.5rem 1fr auto; align-items: center; gap: 0.7rem; margin: 0.6rem 0; }}
.dumbbell-label {{ font-size: 0.74rem; color: var(--ink-dim); text-wrap: balance; }}
.dumbbell-track {{ position: relative; height: 2px; background: var(--line); border-radius: 2px; }}
.dumbbell-line {{ position: absolute; top: 0; height: 2px; background: var(--ink-dim); opacity: 0.55; border-radius: 2px; }}
.dumbbell-dot {{ position: absolute; top: 50%; width: 10px; height: 10px; border-radius: 50%; transform: translate(-50%, -50%); border: 2px solid var(--paper-raised); }}
.dumbbell-dot.dot-a {{ background: var(--cmp-a); z-index: 2; }}
.dumbbell-dot.dot-b {{ background: var(--cmp-b); z-index: 1; }}
.dumbbell-vals {{ display: flex; align-items: center; gap: 0.3rem; font: 0.72rem var(--mono); white-space: nowrap; font-variant-numeric: tabular-nums; }}
.dumbbell-vals .swatch {{ margin-right: -0.05rem; }}
.dumbbell-unit {{ font-style: normal; color: var(--ink-dim); margin-left: 0.15rem; }}
.cf-alt-row {{ padding: 0.6rem 0; border-top: 1px dotted var(--line); }}
.cf-alt-row:first-child {{ border-top: none; }}
.cf-alt-token {{ font: 600 0.86rem var(--mono); margin-right: 0.6rem; }}
.cf-alt-prob {{ font: 0.74rem var(--mono); color: var(--ink-dim); }}
.cf-plot-toggle {{ margin-top: 0.45rem; }}
.cf-plot-toggle > summary {{ list-style: none; cursor: pointer; display: inline-flex; align-items: center; gap: 0.3rem; font: 600 0.72rem var(--sans); color: var(--teal); }}
.cf-plot-toggle > summary::-webkit-details-marker {{ display: none; }}
.cf-plot-toggle > summary::before {{ content: "▸"; font-size: 0.8em; transition: transform 0.15s ease; }}
.cf-plot-toggle[open] > summary::before {{ transform: rotate(90deg); }}
.cf-plot-toggle .scene-traj-img {{ max-width: 22rem; margin-top: 0.5rem; }}
[hidden] {{ display: none !important; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
<div class="page">
  <header class="masthead">
    <span class="eyebrow">Epsilon calibration &middot; same_action noise floor</span>
    <h1>Action-space variance across 100 stochastic rollouts per scene</h1>
    <p class="masthead-meta">
      For each scene, the driving scenario is held fixed and the model is sampled K=100 times. The spread across those
      100 rollouts is the scene's own noise floor: the amount two rollouts can differ by pure chance, before it means
      anything. Every number below is deduplicated against a confirmed OCI Logging Analytics double-delivery artifact --
      one row per real scene.
    </p>
    <div class="masthead-stats">
      <div class="masthead-stat"><b>{total_clips}</b><span>clips</span></div>
      <div class="masthead-stat"><b>{total_scenes}</b><span>scenes</span></div>
      <div class="masthead-stat"><b>{n_categories}</b><span>scenario categories</span></div>
      <div class="masthead-stat"><b>100</b><span>rollouts / scene</span></div>
    </div>
  </header>
  {tab_bar}
  <div class="filter-bar">
    <input id="filter" type="text" placeholder="Filter by clip ID&hellip;" autocomplete="off" />
    <span class="filter-count" id="filter-count"></span>
  </div>
  {panels}
  <footer class="page-footer">
    Amber tick on each gauge marks the recommended epsilon (p90 across scenes in that category) -- the noise level 90%
    of scenes fall under. Teal fill shows where the median sits relative to it. Categories with fewer than 5 clips are
    flagged <em style="color:var(--warn); font-style:normal; font-weight:600;">low-n</em> -- their p90 is closer to a
    max than a percentile and shouldn't be trusted as a stable threshold.
  </footer>
</div>
<script>
(function() {{
  var input = document.getElementById('filter');
  var count = document.getElementById('filter-count');
  var filterBar = document.querySelector('.filter-bar');
  var tabButtons = Array.prototype.slice.call(document.querySelectorAll('[data-tab-btn]'));
  var panels = Array.prototype.slice.call(document.querySelectorAll('[data-tab-panel]'));

  function activePanel() {{
    return document.querySelector('[data-tab-panel]:not([hidden])') || panels[0];
  }}

  function applyFilter() {{
    var panel = activePanel();
    if (!panel) return;
    var q = input.value.trim().toLowerCase();
    var clips = Array.prototype.slice.call(panel.querySelectorAll('details.clip'));
    var clusterSections = Array.prototype.slice.call(panel.querySelectorAll('.cluster-section'));
    var shown = 0;
    clips.forEach(function(clip) {{
      var id = clip.dataset.clipid.toLowerCase();
      var match = q === '' || id.indexOf(q) !== -1;
      clip.hidden = !match;
      if (match) {{ shown++; if (q !== '') clip.open = true; }}
    }});
    clusterSections.forEach(function(section) {{
      var anyVisible = Array.prototype.slice.call(section.querySelectorAll('details.clip')).some(function(c) {{ return !c.hidden; }});
      section.hidden = q !== '' && !anyVisible;
      if (q !== '' && anyVisible) section.querySelector(':scope > details').open = true;
    }});
    count.textContent = q === '' ? '' : (shown + ' match' + (shown === 1 ? '' : 'es'));
  }}

  input.addEventListener('input', applyFilter);

  tabButtons.forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      var tab = btn.dataset.tabBtn;
      tabButtons.forEach(function(b) {{ b.setAttribute('aria-selected', String(b === btn)); }});
      panels.forEach(function(p) {{ p.hidden = p.dataset.tabPanel !== tab; }});
      // The Compare panel has no clip-level drill-down, so the clip-ID
      // filter has nothing to act on there -- hide it rather than let it
      // sit above a view it can't affect.
      if (filterBar) filterBar.hidden = tab === 'compare';
      if (tab !== 'compare') applyFilter();
    }});
  }});
}})();
</script>
"""


def load_videos_b64(video_dir: str | Path) -> dict[str, str]:
    """Reads every {scene_id}.mp4 in video_dir (e.g. produced by
    render_trajectory_overlay.py) and base64-encodes it for inline embedding."""
    import base64

    video_dir = Path(video_dir)
    return {
        p.stem: base64.b64encode(p.read_bytes()).decode("ascii")
        for p in sorted(video_dir.glob("*.mp4"))
    } if video_dir.is_dir() else {}


def load_images_b64(image_dir: str | Path) -> dict[str, str]:
    """Reads every {scene_id}_actions.png in image_dir -- the top-down,
    colored-by-maneuver-class trajectory plot scene_reasoning_report.py
    already writes alongside each scene's reasoning .md, one PNG per scene it
    covers. Unlike load_videos_b64, these files already exist as a normal
    byproduct of fetch_from_logs.py -- there's no extra rendering cost here,
    only the inline-embedding size tradeoff, which is why this is still an
    opt-in flag rather than always-on."""
    import base64

    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return {}
    suffix = "_actions.png"
    return {
        p.name[: -len(suffix)]: base64.b64encode(p.read_bytes()).decode("ascii")
        for p in sorted(image_dir.glob(f"*{suffix}"))
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="pref_pairs/results")
    ap.add_argument("--label", default="Compound noise (reasoning + diffusion)")
    ap.add_argument(
        "--results_dir_b", default=None,
        help="Optional second results dir (e.g. the fixed-reasoning mode's output) -- "
             "rendered as a second tab alongside --results_dir.",
    )
    ap.add_argument("--label_b", default="Diffusion-only noise (reasoning fixed)")
    ap.add_argument("--out", default="pref_pairs/results/noise_floor_report.html")
    ap.add_argument(
        "--video_dir", default=None,
        help="Optional directory of {scene_id}.mp4 files (see render_trajectory_overlay.py) to embed inline.",
    )
    ap.add_argument(
        "--embed_trajectory_pngs", action="store_true",
        help="Embed each dataset's own scene_reasoning/*_actions.png top-down trajectory plots "
             "inline, one per scene. Auto-discovered from --results_dir/--results_dir_b (not a "
             "separate directory flag) since these files are already generated 1:1 with each "
             "results_dir's scene_reasoning reports -- no extra rendering, just embedding size.",
    )
    ap.add_argument(
        "--counterfactual_results_dir", default=None,
        help="Optional counterfactual/results dir (see counterfactual/fetch_from_logs.py) -- "
             "rendered as a 4th 'Token Sensitivity' tab: per-position logit-swap trajectory "
             "deltas, a different experiment (per-token counterfactuals) on the same manifest, "
             "not another noise-floor dataset.",
    )
    ap.add_argument(
        "--counterfactual_example_plots_dir", default=None,
        help="Optional dir of {scene_id}_step{N}_{token}.png files (see "
             "counterfactual/render_examples.py) -- embedded as a toggle button on the "
             "matching alternative row, for the small curated set of examples that have one.",
    )
    args = ap.parse_args()

    data = build_report_data(args.results_dir)
    data_b = build_report_data(args.results_dir_b) if args.results_dir_b else None
    videos = load_videos_b64(args.video_dir) if args.video_dir else {}
    if args.video_dir and not videos:
        logger.warning("--video_dir %s had no .mp4 files", args.video_dir)
    images_a, images_b = {}, {}
    if args.embed_trajectory_pngs:
        images_a = load_images_b64(Path(args.results_dir) / "scene_reasoning")
        if args.results_dir_b:
            images_b = load_images_b64(Path(args.results_dir_b) / "scene_reasoning")

    counterfactual_html = None
    if args.counterfactual_results_dir:
        from counterfactual.render_examples import load_example_plots_b64
        from counterfactual.report import build_counterfactual_data, render_counterfactual_section
        cf_data = build_counterfactual_data(args.counterfactual_results_dir)
        example_plots = (
            load_example_plots_b64(args.counterfactual_example_plots_dir)
            if args.counterfactual_example_plots_dir else {}
        )
        counterfactual_html = render_counterfactual_section(cf_data, example_plots_b64=example_plots)

    html_text = render_html(
        data, args.label, data_b=data_b, label_b=args.label_b, video_b64_by_scene=videos,
        image_b64_by_scene_a=images_a, image_b64_by_scene_b=images_b,
        counterfactual_html=counterfactual_html,
    )
    Path(args.out).write_text(html_text)
    n_tabs = 1 + (2 if data_b else 0) + (1 if counterfactual_html else 0)  # a, [b, compare], [counterfactual]
    logger.info(
        "wrote %s (%d bytes, %d embedded videos, %d+%d embedded trajectory PNGs, %d tab%s)",
        args.out, len(html_text), len(videos), len(images_a), len(images_b),
        n_tabs, "s" if n_tabs != 1 else "",
    )


if __name__ == "__main__":
    main()
