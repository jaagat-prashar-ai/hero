# SPDX-License-Identifier: Apache-2.0
"""Offline select-then-verify analysis over real GRPO rollout dumps.

Consumes the per-scene JSON dumps written by
rl_posttrain/rewards/code_reward_entry.py's `_maybe_dump_rollouts` (opt-in
via CODE_REWARD_DEBUG_DUMP_ROLLOUTS=1, see that module): for each scene
whose clip has a gate-passed cached `reward_fns/<clip_id>.py`, scores every
REAL rollout in the group with that clip's own generated reward function,
takes the argmax, and re-runs the SAME perturbation gate
(code_as_a_reward.clipgen.gate) that validated the function against GT --
this time built from the argmax rollout's own claims and trajectory. This
is the select-then-verify check documented in
code_as_a_reward/clipgen/dossier.py: "the argmax rollout goes through the
same step-5 perturbation battery, built from the rollout itself this time,
only if its corruptions drop does the rollout get trusted" -- it confirms
the argmax is actually good, not just the best of a bad batch.

Nothing here touches the training job or changes what compute_reward_batch
scores; this is entirely offline and read-only against the dumps and the
clip's already-cached reward function.

Per scene, writes:
  - <scene_id>.json     -- every rollout's coc_text/waypoints/trace_reward
                           metrics/clipgen score+components, plus the
                           argmax's full gate-verification result.
  - <scene_id>.overlay.png  -- all rollouts on the t0 camera frame, colored
                           by clipgen score (red->green), argmax thickest.
                           Best-effort: skipped if the frame can't be
                           fetched (e.g. clip outside the warm cache).
Across all scored scenes, writes:
  - heatmap.png          -- clipgen score vs. trace_reward score, per
                           scene, rollouts ranked by clipgen score, with
                           the argmax's own gate pass/fail per row.

Usage:
    python -m code_as_a_reward.clipgen.analyze_group_rollouts \\
        <dump_dir> <reward_fns_dir> <out_dir> [--no-overlays]

<dump_dir> and <reward_fns_dir> may be local paths or s3://bucket/prefix
(the same S3-compat endpoint every other clipgen script uses -- set
AWS_ENDPOINT_URL_S3 / AWS_PROFILE as usual).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from code_as_a_reward.clipgen.gate import GateResult, build_perturbations, run_gate
from code_as_a_reward.clipgen.sandbox import (
    RewardFnError,
    compile_reward_module,
    run_components_fn,
    run_reward_fn,
)
from code_as_a_reward.coc_claim_parser import parse_coc_trace
from pref_pairs.trajectory_features import extract_features


@dataclasses.dataclass
class SelectVerifyResult:
    """Live (non-serialized) result of scoring a rollout group and gate-
    verifying its argmax -- shared by the offline dump analyzer (score_scene,
    which JSON-serializes this) and the generation-time gate loop in
    run_prototype.py (which needs GateResult.feedback() for retries)."""

    scored: list[dict[str, Any]]
    argmax_rollout_id: int | None
    argmax_gate: GateResult | None

_MAX_IMAGE_DIM = 1024


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _split_s3(path: str) -> tuple[str, str]:
    rest = path[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def iter_dump_files(location: str) -> list[tuple[str, bytes]]:
    """[(name, raw_bytes)] for every *.json dump under a local dir or an
    s3://bucket/prefix (recursive -- rollout_dumps/<run_id>/*.json)."""
    if _is_s3(location):
        import boto3

        client = boto3.client("s3")
        bucket, prefix = _split_s3(location)
        out = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                body = client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                out.append((obj["Key"], body))
        return out
    return [(str(p), p.read_bytes()) for p in sorted(Path(location).glob("*.json"))]


def load_reward_source(reward_fns_dir: str, clip_id: str) -> str | None:
    """The clip's cached, gate-passed reward function source, or None if it
    was never generated / never passed the gate (run_prototype.py only
    writes reward_fns/<clip_id>.py on a PASS)."""
    if _is_s3(reward_fns_dir):
        import boto3

        client = boto3.client("s3")
        bucket, prefix = _split_s3(reward_fns_dir)
        key = f"{prefix.rstrip('/')}/{clip_id}.py"
        try:
            return client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        except Exception:
            return None
    p = Path(reward_fns_dir) / f"{clip_id}.py"
    return p.read_text() if p.exists() else None


def select_and_verify(
    clip_id: str, scene_id: str, hz: float, rollouts: list[dict[str, Any]], source: str
) -> SelectVerifyResult:
    """Score every rollout in a group with `source`, take the argmax, then
    build the SAME perturbation battery used at generation time -- but from
    the argmax rollout's own claims and trajectory instead of GT
    (build_perturbations/run_gate take these as plain parameters, no GT
    hardcoded) -- and gate-verify it. This is the select-then-verify step
    from the module docstring, shared by the offline dump analyzer
    (score_scene, below) and run_prototype.py's generation-time gate loop."""
    fn, components_fn = compile_reward_module(source)

    scored: list[dict[str, Any]] = []
    claims_by_rollout: dict[int, Any] = {}
    for ro in rollouts:
        rollout_id = ro["rollout_id"]
        claims = parse_coc_trace(ro["coc_text"], scene_id=clip_id, rollout_id=rollout_id)
        traj = extract_features(ro["waypoints"], hz=hz, scene_id=scene_id, rollout_id=rollout_id)
        claims_by_rollout[rollout_id] = claims
        try:
            clipgen_score = run_reward_fn(fn, claims, traj)
        except RewardFnError:
            clipgen_score = float("nan")
        clipgen_components = None
        if components_fn is not None:
            try:
                clipgen_components = run_components_fn(components_fn, claims, traj)
            except RewardFnError:
                clipgen_components = None
        scored.append({**ro, "clipgen_score": clipgen_score, "clipgen_components": clipgen_components})

    finite = [r for r in scored if np.isfinite(r["clipgen_score"])]
    if not finite:
        return SelectVerifyResult(scored=scored, argmax_rollout_id=None, argmax_gate=None)
    argmax = max(finite, key=lambda r: r["clipgen_score"])
    argmax_id = argmax["rollout_id"]
    argmax_claims = claims_by_rollout[argmax_id]
    argmax_waypoints = np.asarray(argmax["waypoints"], dtype=np.float64)

    cases = build_perturbations(clip_id, argmax_claims, argmax_waypoints, hz, tag=f"argmax_r{argmax_id}")
    gate_result = run_gate(source, cases)
    return SelectVerifyResult(scored=scored, argmax_rollout_id=argmax_id, argmax_gate=gate_result)


def score_scene(dump: dict[str, Any], source: str) -> dict[str, Any]:
    """Score + select + verify, per the module docstring. Returns a fully
    JSON-serializable record (no ParsedCoCTrace/TrajectoryFeatures objects)."""
    clip_id, scene_id, hz = dump["clip_id"], dump["scene_id"], float(dump["hz"])
    result = select_and_verify(clip_id, scene_id, hz, dump["rollouts"], source)
    return {
        "scene_id": scene_id,
        "clip_id": clip_id,
        "hz": hz,
        "rollouts": result.scored,
        "argmax_rollout_id": result.argmax_rollout_id,
        "argmax_gate": None
        if result.argmax_gate is None
        else {
            "passed": result.argmax_gate.passed,
            "pos_score": result.argmax_gate.pos_score,
            "max_pert": result.argmax_gate.max_pert,
            "scores": result.argmax_gate.scores,
            "components": result.argmax_gate.components,
            "failures": result.argmax_gate.failures,
        },
    }


def render_multi_overlay(frame, rollouts: list[dict[str, Any]], cam_intr, cam_extr, argmax_id, max_dim: int = _MAX_IMAGE_DIM):
    """Draw every rollout's trajectory on one frame, colored by
    clipgen_score (RdYlGn: red=low, green=high). The argmax is drawn last
    (so it's never occluded) and thicker."""
    from code_as_a_reward.clipgen.build_overlays import project_waypoints_ftheta
    from matplotlib import colormaps

    cmap = colormaps["RdYlGn"]
    draw = ImageDraw.Draw(frame)
    sx = frame.width / float(cam_intr["width"])
    sy = frame.height / float(cam_intr["height"])
    ordered = sorted(rollouts, key=lambda r: r["rollout_id"] == argmax_id)
    for r in ordered:
        score = r.get("clipgen_score")
        if score is None or not np.isfinite(score):
            continue
        pixels, valid = project_waypoints_ftheta(np.asarray(r["waypoints"], dtype=np.float64), cam_intr, cam_extr)
        pts = [(float(u) * sx, float(v) * sy) for (u, v), ok in zip(pixels, valid) if ok]
        if len(pts) < 2:
            continue
        rgba = cmap(float(np.clip(score, 0.0, 1.0)))
        color = tuple(int(c * 255) for c in rgba[:3])
        is_argmax = r["rollout_id"] == argmax_id
        draw.line(pts, fill=color, width=8 if is_argmax else 4, joint="curve")
    if max(frame.size) > max_dim:
        ratio = max_dim / max(frame.size)
        frame = frame.resize((round(frame.width * ratio), round(frame.height * ratio)), Image.LANCZOS)
    return frame


def render_overlays(records: list[dict[str, Any]], out_dir: Path) -> None:
    """Best-effort: a clip outside the warm-cache mirror (or any fetch
    failure) skips its overlay without aborting the rest of the batch."""
    import boto3

    from code_as_a_reward.clipgen import build_overlays as bo

    client = boto3.client("s3")
    chunk_index = bo._read_parquet_s3(client, f"{bo.WARM_CACHE}/clip_index.parquet")
    calib_cache: dict[int, tuple] = {}
    for record in records:
        clip_id = record["clip_id"]
        try:
            frame, cam_intr, cam_extr = bo.fetch_t0_frame(client, clip_id, chunk_index, calib_cache)
            overlay = render_multi_overlay(frame, record["rollouts"], cam_intr, cam_extr, record["argmax_rollout_id"])
            path = out_dir / f"{record['scene_id']}.overlay.png"
            overlay.save(path)
            print(f"wrote {path}", flush=True)
        except Exception as e:
            print(f"{clip_id}: overlay skipped ({type(e).__name__}: {e})", flush=True)


def build_heatmaps(records: list[dict[str, Any]], out_path: Path) -> None:
    """clipgen score vs. trace_reward's own `reward`, per scene, rollouts
    ranked by clipgen score (argmax first) -- do the two rewards agree on
    which rollout is best? Row labels mark the argmax's own gate verdict."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_max = max(len(r["rollouts"]) for r in records)
    n = len(records)
    clipgen_mat = np.full((n, n_max), np.nan)
    trace_mat = np.full((n, n_max), np.nan)
    row_labels = []
    for i, record in enumerate(records):
        ranked = sorted(
            record["rollouts"],
            key=lambda r: r["clipgen_score"] if np.isfinite(r.get("clipgen_score", np.nan)) else -np.inf,
            reverse=True,
        )
        for j, ro in enumerate(ranked):
            clipgen_mat[i, j] = ro.get("clipgen_score", np.nan)
            trace_mat[i, j] = ro.get("reward", np.nan)
        gate = record.get("argmax_gate")
        mark = "PASS" if gate and gate["passed"] else "FAIL" if gate else "n/a"
        row_labels.append(f"{record['clip_id'][:8]} [{mark}]")

    fig, axes = plt.subplots(1, 2, figsize=(3 + n_max * 0.6, 1.5 + n * 0.5))
    # clipgen_score is sandbox.run_reward_fn's clamped [0, 1] output;
    # trace_reward's `reward` is the unclamped blended reward (roughly
    # [-1, 0.5] in observed runs) -- different scales need different ranges
    # for the colormap to be meaningful on each panel.
    panels = ((axes[0], clipgen_mat, "clipgen score", 0.0, 1.0), (axes[1], trace_mat, "trace_reward score", -1.0, 1.0))
    for ax, mat, title, vmin, vmax in panels:
        im = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_yticks(range(n))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("rollout rank (by clipgen score, argmax = column 0)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze(dump_dir: str, reward_fns_dir: str, out_dir: str, render_images: bool = True) -> list[dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for name, raw in iter_dump_files(dump_dir):
        dump = json.loads(raw)
        source = load_reward_source(reward_fns_dir, dump["clip_id"])
        if source is None:
            print(f"skip {dump['scene_id']}: no cached reward_fns/{dump['clip_id']}.py", flush=True)
            continue
        record = score_scene(dump, source)
        (out / f"{record['scene_id']}.json").write_text(json.dumps(record, indent=2, default=str))
        records.append(record)
        gate = record["argmax_gate"]
        status = "PASS" if gate and gate["passed"] else "FAIL" if gate else "no finite rollout"
        print(f"{record['scene_id']}: argmax=rollout {record['argmax_rollout_id']} gate={status}", flush=True)

    if not records:
        print("no scenes had a cached reward function to score against -- nothing to plot", flush=True)
        return records

    build_heatmaps(records, out / "heatmap.png")
    print(f"wrote {out / 'heatmap.png'}", flush=True)

    if render_images:
        render_overlays(records, out)
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump_dir", help="local dir or s3://bucket/prefix of rollout_dumps/<run_id>/*.json")
    parser.add_argument("reward_fns_dir", help="local dir or s3://bucket/prefix of reward_fns/<clip_id>.py")
    parser.add_argument("out_dir")
    parser.add_argument("--no-overlays", action="store_true", help="skip image overlays (heatmap + JSON only)")
    args = parser.parse_args(argv)
    analyze(args.dump_dir, args.reward_fns_dir, args.out_dir, render_images=not args.no_overlays)


if __name__ == "__main__":
    main()
