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
            cluster_stats[short] = {"median": _median(vals), "p90": _percentile(vals, 0.9)}

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


def render_html(data: dict[str, Any], video_b64_by_scene: dict[str, str] | None = None) -> str:
    """Render build_report_data's output as a single self-contained HTML
    page: gauge cards per category, then a filterable category -> clip ->
    scene drill-down. Design: concrete/asphalt palette, monospace for
    telemetry numbers, serif for reasoning quotes as "transcript" text.

    video_b64_by_scene: optional {scene_id: base64-encoded mp4 bytes} --
    scenes with an entry get an inline <video> (data: URI, so the page stays
    self-contained per the Artifact CSP); scenes without one show no video.
    Deliberately NOT auto-discovered from a directory here -- rendering a
    video is expensive (S3 + log fetches per scene, see
    render_trajectory_overlay.py) and this keeps that cost an explicit
    caller decision, not something that silently scales with report size."""
    video_b64_by_scene = video_b64_by_scene or {}
    cluster_order = sorted(data["clusters"].items(), key=lambda kv: -kv[1]["n_scenes"])
    total_clips = sum(v["n_clips"] for _, v in cluster_order)
    total_scenes = sum(v["n_scenes"] for _, v in cluster_order)

    def gauge_card(cluster: str, v: dict) -> str:
        n = v["n_clips"]
        warn_attr = ' data-lown="1"' if n < 5 else ""
        rows = ""
        for _, (short, label, unit) in METRICS.items():
            s = v["stats"][short]
            med, p90 = s["median"], s["p90"]
            pct = 100.0 if p90 <= 0 else max(6, min(100, (med / p90) * 100))
            rows += (
                f'<div class="gauge-row"><span class="gauge-label">{_esc(label)}</span>'
                f'<div class="gauge-track"><div class="gauge-fill" style="width:{pct:.0f}%"></div>'
                f'<div class="gauge-p90-tick"></div></div>'
                f'<span class="gauge-vals"><b>{_fmt(p90)}</b><i>{_esc(unit)}</i> '
                f'<span class="gauge-med">med {_fmt(med)}</span></span></div>'
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
        video_b64 = video_b64_by_scene.get(scene["scene_id"])
        video_html = (
            f'<video class="scene-video" controls preload="none" playsinline muted loop>'
            f'<source src="data:video/mp4;base64,{video_b64}" type="video/mp4"></video>'
            if video_b64 else ""
        )
        return (
            f'<details class="scene"><summary><span class="scene-t0">t0={_esc(scene["t0_us"])}µs</span>'
            f'<span class="scene-n">{_esc(scene["n_rollouts"])} rollouts</span>{incomplete}</summary>'
            f'<div class="scene-body">{video_html}<div class="stat-row">{stat_html}</div>'
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

    return _PAGE_TEMPLATE.format(
        total_clips=total_clips,
        total_scenes=total_scenes,
        n_categories=len(cluster_order),
        gauge_cards=gauge_cards,
        cluster_sections=cluster_sections,
    )


_PAGE_TEMPLATE = """<title>Action-Space Noise Floor Report</title>
<style>
:root {{
  --paper: #E8E9E4; --paper-raised: #F4F4F0; --ink: #20242B; --ink-dim: #565C63;
  --line: #C9CBC3; --accent: #B96A22; --accent-ink: #FFFFFF; --teal: #2E7671;
  --warn: #A63F35; --good: #4C6A45;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
  --sans: ui-sans-serif, "Neue Haas Grotesk Text", "Helvetica Neue", Arial, sans-serif;
  --serif: Georgia, "Iowan Old Style", "Times New Roman", serif;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --paper: #14171B; --paper-raised: #1B1F24; --ink: #E7E5DD; --ink-dim: #9B9F97;
    --line: #363C42; --accent: #E0954A; --accent-ink: #14171B; --teal: #55A69F;
    --warn: #D9776A; --good: #85A377; }}
}}
:root[data-theme="dark"] {{ --paper: #14171B; --paper-raised: #1B1F24; --ink: #E7E5DD; --ink-dim: #9B9F97;
  --line: #363C42; --accent: #E0954A; --accent-ink: #14171B; --teal: #55A69F; --warn: #D9776A; --good: #85A377; }}
:root[data-theme="light"] {{ --paper: #E8E9E4; --paper-raised: #F4F4F0; --ink: #20242B; --ink-dim: #565C63;
  --line: #C9CBC3; --accent: #B96A22; --accent-ink: #FFFFFF; --teal: #2E7671; --warn: #A63F35; --good: #4C6A45; }}
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
.gauge-row {{ display: grid; grid-template-columns: 6.4rem 1fr auto; align-items: center; gap: 0.6rem; margin: 0.42rem 0; }}
.gauge-label {{ font-size: 0.74rem; color: var(--ink-dim); }}
.gauge-track {{ position: relative; height: 6px; background: var(--line); border-radius: 4px; overflow: visible; }}
.gauge-fill {{ position: absolute; inset: 0; width: 0; background: var(--teal); border-radius: 4px; }}
.gauge-p90-tick {{ position: absolute; right: -1px; top: -3px; width: 2px; height: 12px; background: var(--accent); border-radius: 1px; }}
.gauge-vals {{ font: 0.78rem var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }}
.gauge-vals i {{ font-style: normal; color: var(--ink-dim); margin-right: 0.35rem; }}
.gauge-med {{ color: var(--ink-dim); }}
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
  <section aria-label="Per-category noise floor"><div class="gauge-grid">{gauge_cards}</div></section>
  <div class="filter-bar">
    <input id="filter" type="text" placeholder="Filter by clip ID&hellip;" autocomplete="off" />
    <span class="filter-count" id="filter-count"></span>
  </div>
  <main id="clusters">{cluster_sections}</main>
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
  var clips = Array.prototype.slice.call(document.querySelectorAll('details.clip'));
  var clusterSections = Array.prototype.slice.call(document.querySelectorAll('.cluster-section'));
  input.addEventListener('input', function() {{
    var q = input.value.trim().toLowerCase();
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="pref_pairs/results")
    ap.add_argument("--out", default="pref_pairs/results/noise_floor_report.html")
    ap.add_argument(
        "--video_dir", default=None,
        help="Optional directory of {scene_id}.mp4 files (see render_trajectory_overlay.py) to embed inline.",
    )
    args = ap.parse_args()

    data = build_report_data(args.results_dir)
    videos = load_videos_b64(args.video_dir) if args.video_dir else {}
    if args.video_dir and not videos:
        logger.warning("--video_dir %s had no .mp4 files", args.video_dir)
    html_text = render_html(data, videos)
    Path(args.out).write_text(html_text)
    logger.info("wrote %s (%d bytes, %d embedded videos)", args.out, len(html_text), len(videos))


if __name__ == "__main__":
    main()
