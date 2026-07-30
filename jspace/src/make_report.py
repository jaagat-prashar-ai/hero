"""Build a standalone HTML report from apply_lens + jspace_decomp JSONL dumps.

Reads readouts.jsonl (apply_lens.py) and jspace.jsonl (jspace_decomp.py) and
writes one self-contained HTML file (inline data, no external assets):

  1. per-prompt heatmap, fitted layers x token positions, cell = lens
     probability mass on the <i*> trajectory-token range (README question 2);
     hover shows the cell's top lens tokens, decomposition atoms, and the
     model's own top prediction at that position
  2. J-space explained variance vs depth (gradient-pursuit fit quality)
  3. lens top-1 -> model top-1 agreement vs depth (lens convergence)

Usage:
  python make_report.py --readouts out/readouts.jsonl --jspace out/jspace.jsonl \
      --out out/report.html [--meta "fit: 22 prompts, dim_batch 32"]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_payload(readouts: list[dict], jspace: list[dict], meta: str) -> dict:
    layers = sorted({r["layer"] for r in readouts})
    prompt_idxs = sorted({r["prompt_idx"] for r in readouts})

    atoms = {(r["prompt_idx"], r["layer"], r["pos"]): r["atoms"] for r in jspace}
    ev_by_layer_prompt: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in jspace:
        ev_by_layer_prompt[(r["prompt_idx"], r["layer"])].append(r["explained_var"])

    prompts = []
    for p in prompt_idxs:
        recs = [r for r in readouts if r["prompt_idx"] == p]
        n_pos = max(r["pos"] for r in recs) + 1
        toks = [""] * n_pos
        # traj_mass / tooltip payload per layer x pos
        grid = {l: [None] * n_pos for l in layers}
        cells = {l: [None] * n_pos for l in layers}
        for r in recs:
            l, pos = r["layer"], r["pos"]
            toks[pos] = r["tok"]
            grid[l][pos] = r["traj_mass"]
            cells[l][pos] = {
                "lens": r["lens_top"][:5],
                "model": r["model_top"][:2],
                "atoms": (atoms.get((p, l, pos)) or [])[:5],
            }
        agreement = [
            sum(1 for r in recs if r["layer"] == l and r["lens_top"][0][0] == r["model_top"][0][0])
            / max(sum(1 for r in recs if r["layer"] == l), 1)
            for l in layers
        ]
        ev = [
            (sum(ev_by_layer_prompt[(p, l)]) / len(ev_by_layer_prompt[(p, l)]))
            if ev_by_layer_prompt.get((p, l))
            else None
            for l in layers
        ]
        prompts.append(
            {
                "idx": p,
                "toks": toks,
                "traj": [grid[l] for l in layers],
                "cells": [cells[l] for l in layers],
                "agreement": agreement,
                "explained": ev,
            }
        )
    return {"layers": layers, "prompts": prompts, "meta": meta}


TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Alpamayo 1.5 J-space report</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --ink-1: #0b0b0b; --ink-2: #52514e; --ink-3: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-3: #898781;
    --grid: #2c2c2a; --baseline: #383835; --ring: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d;
  --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-3: #898781;
  --grid: #2c2c2a; --baseline: #383835; --ring: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
}
.viz-root { background: var(--page); color: var(--ink-1);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 24px; }
.viz-root h1 { font-size: 20px; margin: 0 0 4px; }
.viz-root h2 { font-size: 15px; margin: 28px 0 2px; }
.viz-root .sub { color: var(--ink-2); margin: 0 0 10px; font-size: 13px; }
.card { background: var(--surface-1); border: 1px solid var(--ring);
  border-radius: 8px; padding: 14px 16px; margin: 10px 0; }
.scroll { overflow-x: auto; }
.tiles { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
.tile { background: var(--surface-1); border: 1px solid var(--ring);
  border-radius: 8px; padding: 10px 14px; min-width: 130px; }
.tile .lbl { color: var(--ink-2); font-size: 12px; }
.tile .val { font-size: 26px; font-weight: 600; }
.legend { display: flex; gap: 16px; align-items: center; color: var(--ink-2);
  font-size: 12px; margin: 4px 0 8px; flex-wrap: wrap; }
.legend .key { display: inline-flex; align-items: center; gap: 6px; }
.legend .line { width: 18px; height: 2px; display: inline-block; }
#tooltip { position: fixed; pointer-events: none; z-index: 10; display: none;
  background: var(--surface-1); color: var(--ink-1); border: 1px solid var(--ring);
  border-radius: 6px; padding: 8px 10px; font-size: 12px; max-width: 340px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.18); }
#tooltip .row { display: flex; gap: 8px; justify-content: space-between; }
#tooltip .muted { color: var(--ink-2); }
details { margin: 6px 0; color: var(--ink-2); font-size: 12px; }
table { border-collapse: collapse; font-size: 12px; }
td, th { border: 1px solid var(--grid); padding: 2px 8px; text-align: right;
  font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 500; }
svg text { fill: var(--ink-3); font-size: 11px; }
svg .axis { stroke: var(--baseline); stroke-width: 1; }
svg .gridline { stroke: var(--grid); stroke-width: 1; }
</style>
<body class="viz-root">
<h1>Alpamayo 1.5 &middot; J-space quick look</h1>
<p class="sub" id="meta"></p>
<div class="tiles" id="tiles"></div>
<div id="sections"></div>
<div id="tooltip"></div>
<script>
const DATA = __DATA__;
// sequential blue ramp, steps 100..700 (light) — used for the heatmap fill
const RAMP = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7",
              "#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"];
const SERIES = ["var(--s1)","var(--s2)","var(--s3)"];
const tooltip = document.getElementById("tooltip");
const esc = t => { const d = document.createElement("div"); d.textContent = t; return d.innerHTML; };

function showTip(ev, rows) {
  tooltip.innerHTML = "";
  for (const [a, b, strong] of rows) {
    const r = document.createElement("div"); r.className = "row";
    const l = document.createElement("span"); l.className = "muted"; l.textContent = a;
    const v = document.createElement("span"); v.textContent = b;
    if (strong) v.style.fontWeight = "600";
    r.append(l, v); tooltip.append(r);
  }
  tooltip.style.display = "block";
  const pad = 14, w = tooltip.offsetWidth, h = tooltip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tooltip.style.left = x + "px"; tooltip.style.top = y + "px";
}
const hideTip = () => tooltip.style.display = "none";

function heatmap(prompt) {
  const L = DATA.layers, P = prompt.toks.length;
  const cw = 16, ch = 20, left = 46, top = 6, bottom = 74;
  const vmax = Math.max(1e-9, ...prompt.traj.flat().filter(v => v != null));
  const W = left + P * cw + 10, H = top + L.length * ch + bottom;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  for (let i = 0; i < L.length; i++) {
    const lab = document.createElementNS(ns, "text");
    lab.setAttribute("x", left - 6); lab.setAttribute("y", top + i * ch + ch / 2 + 4);
    lab.setAttribute("text-anchor", "end"); lab.textContent = "L" + L[i];
    svg.append(lab);
    for (let p = 0; p < P; p++) {
      const v = prompt.traj[i][p];
      if (v == null) continue;
      const cell = document.createElementNS(ns, "rect");
      cell.setAttribute("x", left + p * cw + 1); cell.setAttribute("y", top + i * ch + 1);
      cell.setAttribute("width", cw - 2); cell.setAttribute("height", ch - 2);
      cell.setAttribute("rx", 2);
      cell.setAttribute("fill", RAMP[Math.min(RAMP.length - 1, Math.floor((v / vmax) * (RAMP.length - 1)))]);
      const c = prompt.cells[i][p];
      cell.addEventListener("pointermove", ev => {
        const rows = [["tok " + p, JSON.stringify(prompt.toks[p]), true],
                      ["layer", "L" + L[i]],
                      ["traj mass", v.toExponential(2), true]];
        c.lens.forEach(([t, pr], j) => rows.push([j ? "" : "lens top", JSON.stringify(t) + "  " + pr.toFixed(3)]));
        c.model.forEach(([t, pr], j) => rows.push([j ? "" : "model", JSON.stringify(t) + "  " + pr.toFixed(3)]));
        c.atoms.forEach(([t, cf], j) => rows.push([j ? "" : "atoms", JSON.stringify(t) + "  " + cf.toFixed(2)]));
        showTip(ev, rows);
      });
      cell.addEventListener("pointerleave", hideTip);
      svg.append(cell);
    }
  }
  for (let p = 0; p < P; p++) {
    const t = document.createElementNS(ns, "text");
    const x = left + p * cw + cw / 2, y = top + L.length * ch + 8;
    t.setAttribute("x", x); t.setAttribute("y", y);
    t.setAttribute("transform", `rotate(-60 ${x} ${y})`);
    t.setAttribute("text-anchor", "end");
    t.textContent = prompt.toks[p].trim() || "\\u00b7";
    svg.append(t);
  }
  return { svg, vmax };
}

function lineChart(series, names, yLabel) {
  const L = DATA.layers;
  const W = 660, H = 240, left = 44, right = 60, top = 12, bottom = 30;
  const iw = W - left - right, ih = H - top - bottom;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  svg.style.maxWidth = "100%";
  const xs = i => left + (i / (L.length - 1)) * iw;
  const ys = v => top + (1 - v) * ih;
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    const ln = document.createElementNS(ns, "line");
    ln.setAttribute("x1", left); ln.setAttribute("x2", left + iw);
    ln.setAttribute("y1", ys(g)); ln.setAttribute("y2", ys(g));
    ln.setAttribute("class", g === 0 ? "axis" : "gridline"); svg.append(ln);
    const t = document.createElementNS(ns, "text");
    t.setAttribute("x", left - 6); t.setAttribute("y", ys(g) + 4);
    t.setAttribute("text-anchor", "end"); t.textContent = g; svg.append(t);
  }
  L.forEach((l, i) => {
    const t = document.createElementNS(ns, "text");
    t.setAttribute("x", xs(i)); t.setAttribute("y", top + ih + 16);
    t.setAttribute("text-anchor", "middle"); t.textContent = "L" + l; svg.append(t);
  });
  series.forEach((vals, s) => {
    const pts = vals.map((v, i) => v == null ? null : [xs(i), ys(v)]).filter(Boolean);
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", pts.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join(" "));
    path.setAttribute("fill", "none"); path.setAttribute("stroke", SERIES[s]);
    path.setAttribute("stroke-width", 2); path.setAttribute("stroke-linejoin", "round");
    path.setAttribute("stroke-linecap", "round"); svg.append(path);
    for (const [x, y] of pts) {
      const ring = document.createElementNS(ns, "circle");
      ring.setAttribute("cx", x); ring.setAttribute("cy", y); ring.setAttribute("r", 6);
      ring.setAttribute("fill", "var(--surface-1)"); svg.append(ring);
      const dot = document.createElementNS(ns, "circle");
      dot.setAttribute("cx", x); dot.setAttribute("cy", y); dot.setAttribute("r", 4);
      dot.setAttribute("fill", SERIES[s]); svg.append(dot);
    }
    const last = pts[pts.length - 1];
    if (last) {
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", last[0] + 10); t.setAttribute("y", last[1] + 4);
      t.style.fill = "var(--ink-2)"; t.textContent = names[s]; svg.append(t);
    }
  });
  // crosshair: snap to nearest layer, one tooltip listing every series
  const hair = document.createElementNS(ns, "line");
  hair.setAttribute("class", "gridline"); hair.setAttribute("y1", top);
  hair.setAttribute("y2", top + ih); hair.style.display = "none"; svg.append(hair);
  const hit = document.createElementNS(ns, "rect");
  hit.setAttribute("x", left); hit.setAttribute("y", top);
  hit.setAttribute("width", iw); hit.setAttribute("height", ih);
  hit.setAttribute("fill", "transparent"); svg.append(hit);
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect();
    const i = Math.max(0, Math.min(L.length - 1,
      Math.round(((ev.clientX - r.left - left) / iw) * (L.length - 1))));
    hair.setAttribute("x1", xs(i)); hair.setAttribute("x2", xs(i));
    hair.style.display = "";
    const rows = [["layer", "L" + L[i], true]];
    series.forEach((vals, s) =>
      rows.push([names[s], vals[i] == null ? "—" : vals[i].toFixed(3)]));
    showTip(ev, rows);
  });
  hit.addEventListener("pointerleave", () => { hair.style.display = "none"; hideTip(); });
  return svg;
}

function legend(names) {
  const div = document.createElement("div"); div.className = "legend";
  names.forEach((n, i) => {
    const k = document.createElement("span"); k.className = "key";
    const sw = document.createElement("span"); sw.className = "line";
    sw.style.background = SERIES[i];
    k.append(sw, document.createTextNode(n)); div.append(k);
  });
  return div;
}

function dataTable(series, names) {
  const det = document.createElement("details");
  const sum = document.createElement("summary"); sum.textContent = "table view";
  det.append(sum);
  const tbl = document.createElement("table");
  const hr = document.createElement("tr");
  hr.append(...["layer", ...names].map(n => { const th = document.createElement("th"); th.textContent = n; return th; }));
  tbl.append(hr);
  DATA.layers.forEach((l, i) => {
    const tr = document.createElement("tr");
    const cells = ["L" + l, ...series.map(s => s[i] == null ? "—" : s[i].toFixed(3))];
    tr.append(...cells.map(c => { const td = document.createElement("td"); td.textContent = c; return td; }));
    tbl.append(tr);
  });
  det.append(tbl);
  return det;
}

const names = DATA.prompts.map(p => "prompt " + p.idx);
document.getElementById("meta").textContent = DATA.meta;
const sections = document.getElementById("sections");

// stat tiles
const allEv = DATA.prompts.flatMap(p => p.explained).filter(v => v != null);
const peak = Math.max(...DATA.prompts.flatMap(p => p.traj.flat()).filter(v => v != null));
const tiles = [["fitted layers", DATA.layers.length],
  ["eval prompts", DATA.prompts.length],
  ["mean J-space var", (allEv.reduce((a, b) => a + b, 0) / allEv.length).toFixed(3)],
  ["peak traj mass", peak.toExponential(1)]];
for (const [lbl, val] of tiles) {
  const d = document.createElement("div"); d.className = "tile";
  const l = document.createElement("div"); l.className = "lbl"; l.textContent = lbl;
  const v = document.createElement("div"); v.className = "val"; v.textContent = val;
  d.append(l, v); document.getElementById("tiles").append(d);
}

// heatmaps
for (const p of DATA.prompts) {
  const h2 = document.createElement("h2");
  h2.textContent = `Trajectory-token mass by layer x position — prompt ${p.idx}`;
  const sub = document.createElement("p"); sub.className = "sub";
  const card = document.createElement("div"); card.className = "card scroll";
  const { svg, vmax } = heatmap(p);
  sub.textContent = `lens probability mass on the <i*> range; fill scaled to this prompt's max (${vmax.toExponential(2)}). Hover a cell for top lens tokens, decomposition atoms, and the model's prediction.`;
  card.append(svg);
  sections.append(h2, sub, card);
}

// line charts
const ev = DATA.prompts.map(p => p.explained);
let h2 = document.createElement("h2");
h2.textContent = "J-space explained variance by depth";
let sub = document.createElement("p"); sub.className = "sub";
sub.textContent = "fraction of residual-stream variance captured by k-sparse nonnegative combinations of J-lens vectors (gradient pursuit), mean over positions";
let card = document.createElement("div"); card.className = "card";
card.append(legend(names), lineChart(ev, names, "explained var"), dataTable(ev, names));
sections.append(h2, sub, card);

const ag = DATA.prompts.map(p => p.agreement);
h2 = document.createElement("h2");
h2.textContent = "Lens top-1 = model top-1 agreement by depth";
sub = document.createElement("p"); sub.className = "sub";
sub.textContent = "how often the layer's J-lens readout already names the model's final next-token prediction";
card = document.createElement("div"); card.className = "card";
card.append(legend(names), lineChart(ag, names, "agreement"), dataTable(ag, names));
sections.append(h2, sub, card);
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readouts", required=True)
    ap.add_argument("--jspace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="")
    args = ap.parse_args()

    payload = build_payload(load_jsonl(args.readouts), load_jsonl(args.jspace), args.meta)
    html = TEMPLATE.replace("__DATA__", json.dumps(payload))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"report -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
