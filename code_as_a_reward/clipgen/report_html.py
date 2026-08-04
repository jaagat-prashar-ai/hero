# SPDX-License-Identifier: Apache-2.0
"""Render a clipgen run's out/ directory into one self-contained report.html.

Everything the run produced is visible on the page -- nothing requires
digging through JSON: the run summary (model, calls, cost, pass rate),
and per clip: the dossier the generator read, the GT reasoning annotation,
every attempt's full transcript (each prompt/reply of the 3-step chain and
retries, collapsible), the generated reward function source, the gate
scorecard (every case's score against its threshold), and the verifier
feedback that drove the next attempt.

Usage:
    python -m code_as_a_reward.clipgen.report_html out_dir [report.html]

No dependencies beyond stdlib; the output opens offline (same pattern as
masking/dashboard). Reads report.json (which since 2026-08-04 carries
per-attempt source + gate_feedback) and transcripts/*.json.
"""

from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

from code_as_a_reward.clipgen.gate import NEG_P95_MAX, POS_MIN

_CSS = """
:root { --bg:#ffffff; --fg:#1a1a1a; --muted:#6b7280; --card:#f6f7f9;
        --border:#d9dce1; --pass:#0a7a33; --passbg:#e3f4e9;
        --fail:#b3261e; --failbg:#fbe9e7; --accent:#1a56db; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#111417; --fg:#e6e8ea; --muted:#9aa1a9; --card:#1a1f24;
          --border:#333a42; --pass:#57c785; --passbg:#12301c;
          --fail:#f28b82; --failbg:#3a1512; --accent:#7aa2f7; }
}
* { box-sizing: border-box; }
body { background:var(--bg); color:var(--fg); margin:0 auto; max-width:1100px;
       padding:24px 16px 80px; font:15px/1.5 system-ui, sans-serif; }
h1 { font-size:1.5em; } h2 { font-size:1.2em; margin-top:2em;
     border-bottom:1px solid var(--border); padding-bottom:6px; }
h3 { font-size:1.0em; margin:1.2em 0 0.4em; }
pre { background:var(--card); border:1px solid var(--border); border-radius:8px;
      padding:12px; overflow-x:auto; font:13px/1.45 ui-monospace, monospace;
      white-space:pre-wrap; word-break:break-word; }
table { border-collapse:collapse; width:100%; margin:8px 0; font-size:14px; }
th, td { border:1px solid var(--border); padding:5px 10px; text-align:left; }
th { background:var(--card); }
.badge { display:inline-block; border-radius:6px; padding:1px 9px;
         font-weight:600; font-size:13px; }
.pass { color:var(--pass); background:var(--passbg); }
.fail { color:var(--fail); background:var(--failbg); }
.muted { color:var(--muted); }
.kpis { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }
.kpi { background:var(--card); border:1px solid var(--border); border-radius:10px;
       padding:10px 16px; min-width:130px; }
.kpi b { display:block; font-size:1.25em; }
details { margin:8px 0; }
summary { cursor:pointer; color:var(--accent); font-weight:600; }
details > div { border-left:3px solid var(--border); margin-left:4px;
                padding-left:12px; }
.role { font-weight:700; margin-top:10px; }
"""


def _e(s: object) -> str:
    return html.escape(str(s))


def _fmt(x: object) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "NaN (raised)"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _badge(ok: bool, label_ok: str = "PASS", label_bad: str = "FAIL") -> str:
    cls, label = ("pass", label_ok) if ok else ("fail", label_bad)
    return f'<span class="badge {cls}">{label}</span>'


def _score_table(scores: dict[str, float], gate_cases: list[dict]) -> str:
    kinds = {c["name"]: c["kind"] for c in gate_cases}
    rows = []
    for name, score in scores.items():
        kind = kinds.get(name, "?")
        if kind == "positive":
            ok = isinstance(score, float) and math.isfinite(score) and score >= POS_MIN
            want = f"&ge; {POS_MIN}"
        else:
            ok = isinstance(score, float) and math.isfinite(score) and score <= NEG_P95_MAX
            want = f"&le; {NEG_P95_MAX}"
        rows.append(
            f"<tr><td>{_e(name)}</td><td>{_e(kind)}</td>"
            f"<td>{_fmt(score)}</td><td>{want}</td><td>{_badge(ok, 'ok', 'violates')}</td></tr>"
        )
    return (
        "<table><tr><th>gate case</th><th>kind</th><th>score</th>"
        "<th>required</th><th>verdict</th></tr>" + "".join(rows) + "</table>"
    )


def _transcript_html(transcript_path: Path) -> str:
    if not transcript_path.exists():
        return '<p class="muted">no transcript file</p>'
    turns = json.loads(transcript_path.read_text())
    parts = []
    for turn in turns:
        role = turn.get("role", "?")
        content = turn.get("content", "")
        if not isinstance(content, str):  # anthropic block lists
            content = json.dumps(content, indent=1)
        parts.append(f'<div class="role">{_e(role)}</div><pre>{_e(content)}</pre>')
    return "".join(parts)


def render(out_dir: str | Path) -> str:
    out = Path(out_dir)
    report = json.loads((out / "report.json").read_text())
    clips = report.get("clips", {})
    n_pass = sum(1 for e in clips.values() if e.get("passed"))

    body = [f"<h1>clipgen run report <span class='muted'>({_e(out)})</span></h1>"]
    body.append('<div class="kpis">')
    for label, value in [
        ("model", report.get("model") or "dry-run"),
        ("clips passed", f"{n_pass}/{len(clips)}"),
        ("success bar", "&ge; 4/5"),
        ("API calls", report.get("api_calls", 0)),
        ("cost", f"${report.get('api_cost_usd', 0):.2f}"),
    ]:
        body.append(f'<div class="kpi"><b>{value}</b><span class="muted">{label}</span></div>')
    body.append("</div>")
    if report.get("aborted"):
        body.append(f'<p>{_badge(False, "", "ABORTED")} {_e(report["aborted"])}</p>')

    for clip_id, entry in clips.items():
        body.append(f"<h2>clip {_e(clip_id)} {_badge(entry.get('passed', False))}</h2>")
        if entry.get("gt_coc"):
            body.append(f"<h3>ground-truth reasoning</h3><pre>{_e(entry['gt_coc'])}</pre>")
        dossier = out / f"{clip_id}.dossier.txt"
        if dossier.exists():
            body.append(
                "<details><summary>dossier (what the generator read)</summary>"
                f"<div><pre>{_e(dossier.read_text())}</pre></div></details>"
            )
        gate_cases = entry.get("gate_cases", [])
        if gate_cases:
            inventory = ", ".join(f"{c['name']}" for c in gate_cases)
            body.append(
                f"<p class='muted'>{entry.get('n_gate_cases', len(gate_cases))} gate cases: "
                f"{_e(inventory)}</p>"
            )

        for att in entry.get("attempts", []):
            n = att.get("attempt", "?")
            if "error" in att:
                body.append(
                    f"<h3>attempt {n} {_badge(False, '', 'ERROR')}</h3>"
                    f"<pre>{_e(att['error'])}</pre>"
                )
                continue
            body.append(
                f"<h3>attempt {n} {_badge(att.get('passed', False))} "
                f"<span class='muted'>pos {_fmt(att.get('pos_score'))} "
                f"(need &ge; {POS_MIN}) · neg p95 {_fmt(att.get('neg_p95'))} "
                f"(need &le; {NEG_P95_MAX})</span></h3>"
            )
            if att.get("source"):
                body.append(
                    "<details open><summary>generated reward function</summary>"
                    f"<div><pre>{_e(att['source'])}</pre></div></details>"
                )
            if att.get("scores"):
                body.append(_score_table(att["scores"], gate_cases))
            if att.get("gate_feedback"):
                body.append(
                    "<details open><summary>verifier feedback (sent to next attempt)"
                    f"</summary><div><pre>{_e(att['gate_feedback'])}</pre></div></details>"
                )
            body.append(
                f"<details><summary>full transcript (attempt {n})</summary><div>"
                + _transcript_html(out / "transcripts" / f"{clip_id}.attempt{n}.json")
                + "</div></details>"
            )
        if not entry.get("attempts"):
            body.append('<p class="muted">no attempts (dry run)</p>')

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>clipgen report — {_e(report.get('summary', ''))}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<p class='muted'>{_e(report.get('summary', ''))}</p>" + "".join(body) + "</body></html>"
    )


if __name__ == "__main__":
    out_dir = sys.argv[1]
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(out_dir) / "report.html"
    target.write_text(render(out_dir))
    print(f"wrote {target}")
