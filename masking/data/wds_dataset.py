# SPDX-License-Identifier: Apache-2.0
"""
wds_dataset.py — WebDataset pipeline that converts WDS shards into the sample
format expected by the masking experiments.

Each WDS sample contains (per clip):
  {clip_id}.json           — clip metadata (clip_id, events, sampled_t0_us, …)
  {clip_id}.ego.npz        — full clip egomotion (timestamps_us, xyz, velocity, curvature)
  {clip_id}.snapshots.npz  — per-snapshot sensor data:
                               t0_us        (S,)
                               frames       (S, 4_cams, 4_frames, H, W, 3) uint8
                               ego_hist_xyz (S, 16, 3) float32
                               ego_fut_xyz  (S, 64, 3) float32
                               traj_none_xyz  (S, 63, 3) float32
                               traj_mask_xyz  (S, 63, 3) float32
                               cot          (S,) str

For the masking experiments each snapshot is expanded into one item with:
  clip_id:        str
  t0_us:          int
  event_cluster:  str  (from metadata JSON, if available)
  group:          str  (from metadata JSON)
  was_clamped:    bool
  model_inputs:   dict compatible with build_inputs() / run_masked_openloop.py
    image_frames:     (4, 4, 3, H, W) uint8 torch.Tensor
    camera_indices:   (4,) torch.Tensor — fixed [0, 1, 2, 6] for the 4-cam rig
    ego_history_xyz:  (1, 1, 16, 3) float32 torch.Tensor
    ego_history_rot:  (1, 1, 16, 3, 3) float32 torch.Tensor  [identity — see NOTE]
    ego_future_xyz:   (1, 1, 64, 3) float32 torch.Tensor

NOTE on ego_history_rot
-----------------------
The WDS build step (build_webdataset.py) saves ego_hist_xyz but not ego_history_rot
because only xyz is needed to reproduce the ADE comparison between masked and
unmasked conditions. Since BOTH conditions use the same rotation, the *relative*
delta between them is unaffected. We use identity rotation matrices as a
placeholder here so the model's action_to_traj call produces world-frame
trajectories in an approximate sensor frame. If absolute world-frame accuracy is
needed, rerun the WDS build with rotation data included.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Generator, Iterable

import numpy as np
import torch
import webdataset as wds

logger = logging.getLogger(__name__)

# Camera indices for the standard 4-cam rig (CROSS_LEFT, FRONT_WIDE, CROSS_RIGHT, FRONT_TELE)
CAMERA_INDICES = torch.tensor([0, 1, 2, 6], dtype=torch.long)


def _decode_npz(data: bytes) -> dict[str, np.ndarray]:
    """Load a .npz file from raw bytes."""
    return dict(np.load(io.BytesIO(data), allow_pickle=True))


def _decode_json(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


def _identity_rotations(n_hist: int = 16) -> torch.Tensor:
    """Return (1, 1, n_hist, 3, 3) identity rotation matrices."""
    eye = torch.eye(3, dtype=torch.float32)
    return eye.view(1, 1, 1, 3, 3).expand(1, 1, n_hist, 3, 3).contiguous()


def _expand_clip_to_snapshots(sample: dict) -> list[dict]:
    """
    Expand one WDS clip sample into one dict per snapshot.

    Args:
        sample: raw WDS dict with keys '__key__', 'json', 'ego.npz', 'snapshots.npz'

    Returns:
        List of per-snapshot dicts, each with model_inputs ready for build_inputs().
    """
    clip_id = sample["__key__"]

    # ── Decode payloads ─────────────────────────────────────────────────────────
    try:
        meta = _decode_json(sample["json"])
    except Exception as exc:
        logger.warning("clip %s: failed to decode json: %s", clip_id, exc)
        return []

    try:
        snaps = _decode_npz(sample["snapshots.npz"])
    except Exception as exc:
        logger.warning("clip %s: failed to decode snapshots.npz: %s", clip_id, exc)
        return []

    # ── Extract snapshot arrays ─────────────────────────────────────────────────
    # frames: (S, 4, 4, H, W, 3) uint8
    # ego_hist_xyz: (S, 16, 3) float32
    # ego_fut_xyz:  (S, 64, 3) float32
    # t0_us: (S,) int64
    frames_all: np.ndarray = snaps["frames"]          # (S, 4, 4, H, W, 3)
    ego_hist: np.ndarray   = snaps["ego_hist_xyz"]    # (S, 16, 3)
    ego_fut: np.ndarray    = snaps["ego_fut_xyz"]     # (S, 64, 3)
    t0_us_arr: np.ndarray  = snaps["t0_us"]           # (S,)

    S = int(t0_us_arr.shape[0])

    # ── Build event metadata lookup for this clip ───────────────────────────────
    # events list from the JSON: [{t0_us, event_cluster, group, was_clamped, …}, …]
    events_by_t0: dict[int, dict] = {}
    for ev in meta.get("events", []):
        events_by_t0[int(ev["t0_us"])] = ev

    # ── One output item per snapshot ────────────────────────────────────────────
    items: list[dict] = []
    for s_idx in range(S):
        t0_us = int(t0_us_arr[s_idx])
        ev    = events_by_t0.get(t0_us, {})

        # frames[s_idx]: (4_cams, 4_frames, H, W, 3) uint8
        # Convert to (4, 4, 3, H, W) as expected by the model
        frame_np = frames_all[s_idx]                  # (4, 4, H, W, 3)
        frame_t  = torch.from_numpy(
            np.ascontiguousarray(frame_np.transpose(0, 1, 4, 2, 3))
        )                                              # (4, 4, 3, H, W)

        # ego history/future: add batch + group dimensions → (1, 1, N, 3)
        h_xyz = torch.from_numpy(ego_hist[s_idx]).float().unsqueeze(0).unsqueeze(0)
        f_xyz = torch.from_numpy(ego_fut[s_idx]).float().unsqueeze(0).unsqueeze(0)

        n_hist = h_xyz.shape[2]
        rot    = _identity_rotations(n_hist)  # (1, 1, 16, 3, 3)

        items.append({
            "clip_id":       clip_id,
            "t0_us":         t0_us,
            "event_cluster": ev.get("event_cluster", "UNKNOWN"),
            "group":         ev.get("group", "unknown"),
            "was_clamped":   bool(ev.get("was_clamped", False)),
            "event_idx":     int(ev.get("event_idx", 0)),
            "event_coc":     str(ev.get("event_coc", "")),
            "model_inputs": {
                "image_frames":    frame_t,          # (4, 4, 3, H, W)
                "camera_indices":  CAMERA_INDICES,   # (4,)
                "ego_history_xyz": h_xyz,            # (1, 1, 16, 3)
                "ego_history_rot": rot,              # (1, 1, 16, 3, 3)
                "ego_future_xyz":  f_xyz,            # (1, 1, 64, 3)
            },
        })
    return items


def _no_node_split(src, group=None):
    """No-op nodesplitter: every rank sees every shard.

    wds.WebDataset defaults to wds.single_node_only, which *raises* under
    multi-node (world_size > 1) training unless an explicit nodesplitter is
    given. We must pass one, but it must NOT slice the shard list by rank —
    run.py's masking_loop already partitions at the sample level via
    _shard_owner() across the FULL shard list given to every rank.
    """
    yield from src


def iter_snapshots(
    shard_paths: Iterable[str | Path],
    *,
    shuffle_shards: bool = False,
    resampled: bool = False,
) -> Generator[dict, None, None]:
    """
    Yield one snapshot item per sample from the provided WDS shard paths.

    Args:
        shard_paths: Local .tar paths or s3:// URLs understood by webdataset.
        shuffle_shards: Shuffle shard order (useful for training; off for inference).
        resampled: Use wds.ResampledShards for infinite looping (training only).

    Yields:
        Dicts with keys: clip_id, t0_us, event_cluster, group, was_clamped,
        event_idx, event_coc, model_inputs.
    """
    str_paths = [str(p) for p in shard_paths]
    if not str_paths:
        logger.warning("iter_snapshots: no shard paths provided")
        return

    # nodesplitter=_no_node_split (not wds.split_by_node): callers (run.py's
    # masking_loop) already give every rank the FULL shard list and partition
    # at the sample level via _shard_owner() hashing. wds.split_by_node would
    # additionally slice the shard list itself by torch.distributed
    # rank/world_size, which silently drops every shard for any rank >=
    # len(shards) (e.g. 2 shards, 8 ranks -> "No samples found in dataset;
    # perhaps you have fewer shards than workers.").
    if resampled:
        dataset = wds.WebDataset(
            wds.ResampledShards(str_paths),
            shardshuffle=shuffle_shards,
            nodesplitter=_no_node_split,
        )
    else:
        dataset = wds.WebDataset(
            str_paths,
            shardshuffle=shuffle_shards,
            nodesplitter=_no_node_split,
        )

    # Decode the fields we need; leave mp4 out (not used for inference)
    dataset = dataset.decode()

    for raw_sample in dataset:
        # Normalize key names: webdataset decoder strips the base key so we
        # get e.g. {"__key__": "clip_id", "json": b"...", "ego.npz": b"...", ...}
        # Re-encode the already-decoded payload back to bytes for _expand_clip_to_snapshots
        # Actually wds.decode() returns numpy for npz and str for json — handle both.
        normalized: dict = {"__key__": raw_sample["__key__"]}

        # json may already be decoded to a dict or remain bytes
        if "json" in raw_sample:
            v = raw_sample["json"]
            normalized["json"] = json.dumps(v).encode() if isinstance(v, dict) else v
        if "ego.npz" in raw_sample:
            normalized["ego.npz"] = raw_sample["ego.npz"]
        if "snapshots.npz" in raw_sample:
            normalized["snapshots.npz"] = raw_sample["snapshots.npz"]

        # Handle partially decoded data: if wds decoded npz to dict of arrays,
        # re-pack it for _decode_npz; if it decoded json to dict, pass through.
        try:
            items = _expand_clip_to_snapshots_from_decoded(raw_sample)
        except Exception as exc:
            logger.warning("Error expanding clip %s: %s", raw_sample.get("__key__"), exc)
            continue

        yield from items


def _expand_clip_to_snapshots_from_decoded(sample: dict) -> list[dict]:
    """
    Same as _expand_clip_to_snapshots but handles the mixed decoded/raw
    output that wds.decode() returns:
      - 'json' may be a dict (already decoded by wds) or bytes
      - 'ego.npz' / 'snapshots.npz' may be a dict of arrays (decoded) or bytes
    """
    clip_id = sample["__key__"]

    # ── Decode JSON ──────────────────────────────────────────────────────────────
    raw_json = sample.get("json", {})
    if isinstance(raw_json, bytes):
        meta = json.loads(raw_json.decode("utf-8"))
    elif isinstance(raw_json, str):
        meta = json.loads(raw_json)
    else:
        meta = raw_json  # already a dict

    # ── Decode snapshots.npz ─────────────────────────────────────────────────────
    raw_snaps = sample.get("snapshots.npz")
    if raw_snaps is None:
        logger.warning("clip %s: missing snapshots.npz", clip_id)
        return []
    if isinstance(raw_snaps, bytes):
        snaps = dict(np.load(io.BytesIO(raw_snaps), allow_pickle=True))
    elif isinstance(raw_snaps, dict):
        snaps = raw_snaps
    else:
        # wds may return an NpzFile object
        snaps = dict(raw_snaps)

    # ── Extract arrays ───────────────────────────────────────────────────────────
    frames_all: np.ndarray = np.asarray(snaps["frames"])       # (S, 4, 4, H, W, 3)
    ego_hist: np.ndarray   = np.asarray(snaps["ego_hist_xyz"]) # (S, 16, 3)
    ego_fut: np.ndarray    = np.asarray(snaps["ego_fut_xyz"])  # (S, 64, 3)
    t0_us_arr: np.ndarray  = np.asarray(snaps["t0_us"])        # (S,)

    S = int(t0_us_arr.shape[0])

    events_by_t0: dict[int, dict] = {}
    for ev in meta.get("events", []):
        events_by_t0[int(ev["t0_us"])] = ev

    items: list[dict] = []
    for s_idx in range(S):
        t0_us = int(t0_us_arr[s_idx])
        ev    = events_by_t0.get(t0_us, {})

        frame_np = np.asarray(frames_all[s_idx])              # (4, 4, H, W, 3) uint8
        frame_t  = torch.from_numpy(
            np.ascontiguousarray(frame_np.transpose(0, 1, 4, 2, 3))
        )                                                       # (4, 4, 3, H, W)

        h_xyz = torch.from_numpy(np.asarray(ego_hist[s_idx], dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        f_xyz = torch.from_numpy(np.asarray(ego_fut[s_idx], dtype=np.float32)).unsqueeze(0).unsqueeze(0)

        n_hist = h_xyz.shape[2]
        rot    = _identity_rotations(n_hist)

        items.append({
            "clip_id":       clip_id,
            "t0_us":         t0_us,
            "event_cluster": ev.get("event_cluster", "UNKNOWN"),
            "group":         ev.get("group", "unknown"),
            "was_clamped":   bool(ev.get("was_clamped", False)),
            "event_idx":     int(ev.get("event_idx", 0)),
            "event_coc":     str(ev.get("event_coc", "")),
            "model_inputs": {
                "image_frames":    frame_t,
                "camera_indices":  CAMERA_INDICES,
                "ego_history_xyz": h_xyz,
                "ego_history_rot": rot,
                "ego_future_xyz":  f_xyz,
            },
        })
    return items
