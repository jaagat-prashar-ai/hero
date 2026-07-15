# On-cluster clip sampling + extraction, so the Lilypad job is fully
# self-contained -- no locally pre-built manifest or clip cache. Reuses
# masking.data.sample_clips's existing shard-scanning machinery
# (find_available_ood_clips, load_ood_clips, load_feature_presence,
# filter_clips_with_required_features) rather than reimplementing it, and
# adds the one missing piece: walking forward from a clip's known tar
# group-start offset to pull its actual egomotion.parquet + camera mp4
# bytes -- scan_shard_for_clips (in that module) only extracts the trailing
# .json metadata member, deliberately skipping over the multi-MB payloads.
#
# sample_and_resolve_clips is deterministic given (n_per_cluster, seed): every
# rank independently recomputes the SAME full clip list (no cross-rank
# coordination/broadcast needed) and then filters down to its own shard --
# same pattern training/run.py already uses for rank sharding.

import logging
import os

import boto3
import numpy as np

from masking.data.sample_clips import (
    TAR_BLOCK,
    filter_clips_with_required_features,
    find_available_ood_clips,
    load_feature_presence,
    load_ood_clips,
)

logger = logging.getLogger(__name__)

BUCKET = "research-datasets-chicago"
PREFIX = "nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/train"
CAMERA_KEYS = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)
# The MIN_T0_US threshold for t0 validity lives in sample_scenario_clips.py
# (single source of truth) -- pick_t0_and_coc, reused below, applies it.


def sample_and_resolve_clips(n_per_cluster: int, seed: int, hf_token: str | None) -> list[dict]:
    """Pick n_per_cluster clips per event_cluster (from what's actually in S3) + resolve t0_us/coc."""
    # Imported lazily: sample_and_resolve_clips only ever runs on rank 0 in
    # the cluster's BASE Python 3.10 env (via training/run.py), while this
    # module is ALSO flat-imported by cluster_worker.py inside the isolated
    # Python 3.12 alpamayo venv (for extract_clip_to_dir) -- keeping this
    # import out of module scope keeps the worker's import surface unchanged.
    from perplexity.sample_scenario_clips import pick_t0_and_coc

    ood_df = load_ood_clips(hf_token)
    feature_df = load_feature_presence(hf_token)
    ood_df = filter_clips_with_required_features(ood_df, feature_df)

    by_cluster = find_available_ood_clips(BUCKET, PREFIX, ood_df, n_per_type=n_per_cluster)

    rng = np.random.default_rng(seed)
    resolved: list[dict] = []
    for cluster in sorted(by_cluster):
        candidates = sorted(by_cluster[cluster], key=lambda c: c[0])  # clip_id, for determinism
        order = rng.permutation(len(candidates))
        picked = 0
        for i in order:
            if picked >= n_per_cluster:
                break
            clip_id, shard_key, _meta, offset = candidates[i]
            # t0_us/coc MUST come from ood_reasoning.parquet's own "events"
            # column (same convention as sample_scenario_clips.py /
            # sample_ood_clips.py), NOT from the WDS shard's per-clip json
            # blob (_meta). That blob has no "events" key at all (verified
            # live against shard_016_00046.tar: its keys are clip_id /
            # collection / feature_presence / ood_events / video), so the
            # previous meta.get("events") lookup here silently rejected
            # EVERY candidate -- canary3 resolved 0/1 clips for all 9
            # clusters because of this.
            t0_and_coc = pick_t0_and_coc(ood_df, clip_id)
            if t0_and_coc is None:
                continue  # no event clears MIN_T0_US -- draw a different candidate
            t0_us, coc = t0_and_coc
            resolved.append(
                {
                    "clip_id": clip_id,
                    "event_cluster": cluster,
                    "shard_key": shard_key,
                    "group_start_offset": offset,
                    "t0_us": t0_us,
                    "coc": coc,
                }
            )
            picked += 1
        if picked < n_per_cluster:
            logger.warning(
                "event_cluster=%s: only %d/%d clips resolved (not enough S3-available "
                "candidates with a valid t0_us)",
                cluster,
                picked,
                n_per_cluster,
            )
    return resolved


def _range_get(s3, key: str, start: int, length: int) -> bytes:
    end = start + length - 1
    return s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={start}-{end}")["Body"].read()


def extract_clip_to_dir(shard_key: str, group_start_offset: int, clip_id: str, dest_dir: str) -> str:
    """Walk shard_key's tar forward from group_start_offset, pulling this clip's
    egomotion.parquet + 4 camera mp4s into dest_dir, named exactly as
    s3_clip_loader.load_clip_from_s3_extract expects
    ({clip_id}.egomotion.parquet, {clip_id}.camera_<key>.mp4).

    Same 512-byte tar-header convention as
    masking.data.sample_clips.scan_shard_for_clips (name at hdr[0:100], octal
    size at hdr[124:136]) -- walks the same per-clip file group that
    function skips over (it only extracts the trailing .json member),
    stopping once the clip_id prefix changes (next clip's group starts).
    """
    os.makedirs(dest_dir, exist_ok=True)
    wanted = {"egomotion.parquet"} | {f"{k}.mp4" for k in CAMERA_KEYS}

    s3 = boto3.client("s3")
    pos = group_start_offset
    zero_blocks = 0
    found: set[str] = set()
    while True:
        hdr = _range_get(s3, shard_key, pos, TAR_BLOCK)
        if len(hdr) < TAR_BLOCK or hdr == b"\x00" * TAR_BLOCK:
            zero_blocks += 1
            pos += TAR_BLOCK
            if zero_blocks >= 2:
                break
            continue
        zero_blocks = 0
        name = hdr[0:100].split(b"\x00")[0].decode(errors="replace")
        size_field = hdr[124:136].split(b"\x00")[0].strip()
        size = int(size_field, 8) if size_field else 0
        data_start = pos + TAR_BLOCK
        is_pax = "@PaxHeader" in name

        if not is_pax:
            this_clip_id = name.split(".", 1)[0] if "." in name else name
            if this_clip_id != clip_id:
                break  # walked into the next clip's group -- this one is done
            suffix = name[len(clip_id) + 1 :] if name.startswith(clip_id + ".") else None
            if suffix in wanted:
                data = _range_get(s3, shard_key, data_start, size)
                with open(os.path.join(dest_dir, name), "wb") as f:
                    f.write(data)
                found.add(suffix)
                if found >= wanted:
                    break

        data_blocks = (size + TAR_BLOCK - 1) // TAR_BLOCK
        pos = data_start + data_blocks * TAR_BLOCK

    missing = wanted - found
    if missing:
        raise RuntimeError(f"clip {clip_id}: missing files in shard {shard_key}: {missing}")
    return dest_dir
