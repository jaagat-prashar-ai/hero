"""Extract short front-camera preview clips (mp4) for a handful of sample
events, for the dashboard to play alongside each clip's trajectory result.

Pulls only the needed clip's tar members via S3 range-reads (see
masking.data.s3_clip_extract), decodes a window of frames around t0 with
PyAV, and re-encodes a compact preview mp4 -- no full shard download.

Requires AWS credentials/endpoint for the OCI S3-compatible bucket in the
environment (e.g. AWS_PROFILE=oci.chi, AWS_ENDPOINT_URL_S3=...).

Usage:
    python3 masking/dashboard/generate_video_previews.py \
        --a batch_experiment_a.jsonl --manifest masking/configs/sample_clips.json \
        --outdir masking/dashboard/videos --n 5
"""
from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av

from masking.data.s3_clip_extract import extract_clip_members

BUCKET = "research-datasets-chicago"
CAMERA = "camera_front_wide_120fov.mp4"
WINDOW_BEFORE_S = 1.6   # matches NUM_HISTORY_STEPS * TIME_STEP_S
WINDOW_AFTER_S = 6.4    # matches NUM_FUTURE_STEPS * TIME_STEP_S
OUT_WIDTH = 640


def pick_clips(a_rows: list[dict], n: int) -> list[dict]:
    ranked = sorted(a_rows, key=lambda r: -r["ade_m"])
    if n <= 2:
        return ranked[:n]
    top = max(1, n - 2)
    picks = ranked[:top] + [ranked[len(ranked) // 2], ranked[-1]]
    seen, out = set(), []
    for r in picks:
        if r["clip_id"] not in seen:
            seen.add(r["clip_id"])
            out.append(r)
    return out[:n]


def make_preview(row: dict, shard_key: str, offset: int, outdir: Path) -> dict:
    clip_id, t0_us = row["clip_id"], row["t0_us"]
    short = clip_id[:8]
    members = extract_clip_members(BUCKET, shard_key, clip_id, offset)
    mp4_bytes = members[CAMERA]

    in_container = av.open(io.BytesIO(mp4_bytes))
    in_stream = in_container.streams.video[0]
    fps = float(in_stream.average_rate)
    t0_s = t0_us / 1e6
    start_s = max(0.0, t0_s - WINDOW_BEFORE_S)
    end_s = t0_s + WINDOW_AFTER_S

    out_path = outdir / f"{short}.mp4"
    out_container = av.open(str(out_path), mode="w")
    out_stream = out_container.add_stream("h264", rate=int(round(fps)))
    out_stream.width = OUT_WIDTH
    out_stream.height = int(round(in_stream.height * OUT_WIDTH / in_stream.width))
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "28", "preset": "veryfast"}

    time_base = float(in_stream.time_base)
    in_container.seek(int(start_s / time_base), stream=in_stream, any_frame=False, backward=True)
    n_frames = 0
    for frame in in_container.decode(in_stream):
        if frame.time is None or frame.time < start_s:
            continue
        if frame.time > end_s:
            break
        frame = frame.reformat(width=out_stream.width, height=out_stream.height)
        for packet in out_stream.encode(frame):
            out_container.mux(packet)
        n_frames += 1
    for packet in out_stream.encode():
        out_container.mux(packet)
    out_container.close()
    in_container.close()

    return {
        "clip_id": clip_id, "clip_id_short": short, "event_cluster": row["event_cluster"],
        "ade_m": row["ade_m"], "t0_us": t0_us, "video": out_path.name,
        "window_start_s": round(start_s - t0_s, 2), "window_end_s": round(end_s - t0_s, 2),
        "n_frames": n_frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", default="masking/dashboard/videos")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    with open(args.a) as fh:
        a_rows = [json.loads(l) for l in fh]
    with open(args.manifest) as fh:
        manifest = {r["clip_id"]: r for r in json.load(fh)}

    picks = pick_clips(a_rows, args.n)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=len(picks)) as ex:
        futs = {}
        for r in picks:
            m = manifest[r["clip_id"]]
            fut = ex.submit(make_preview, r, m["shard_key"], m["offset"], outdir)
            futs[fut] = r["clip_id"]
        for fut in as_completed(futs):
            clip_id = futs[fut]
            try:
                info = fut.result()
                results.append(info)
                print(f"done: {clip_id} -> {info['video']} ({info['n_frames']} frames)")
            except Exception as exc:
                print(f"FAILED: {clip_id}: {exc}")

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} previews to {outdir}/, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
