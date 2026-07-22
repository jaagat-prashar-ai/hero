# SPDX-License-Identifier: Apache-2.0
"""Select the N PAI chunks densest in OOD-reasoning clips and write the
mini clip index for them.

Why this exists: the vendored scripts/download_pai.py --only-reasoning-chunks
path RANDOMLY samples clips and then downloads every chunk any sampled clip
touches. The 1740 OOD clips are scattered ~1.6-per-chunk across 1085 chunks
(measured 2026-07-22), so random sampling costs ~5.7 GB of camera download
per clip. Picking the OOD-DENSEST chunks instead roughly doubles
clips-per-GB (densest 100 chunks -> 394 clips vs ~180 for random).

Runs inside the recipe venv (huggingface_hub + pandas guaranteed -- the
vendored download_pai.py itself needs both). Outputs into --output-dir:
  - clip_index_reasoning_mini.parquet  -- the exact artifact download_pai's
    --num-reasoning-clips mode writes (clip_index rows for the selected
    clips, original index dtype preserved), which the RL entry's hydra
    override reads as data.train.dataset.clip_index_metadata.
  - dense_chunks.txt -- space-separated chunk IDs, the exact string
    scripts/download_pai.py --chunk-ids parses.

The caller (run.py::_download_pai_reasoning_dense) then invokes the vendored
downloader with --chunk-ids from that file; this script itself downloads
only the two small metadata parquets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"

# Constants mirroring the recipe runtime (src/alpamayo/data/pai_utils.py and
# the alpamayo1_5_rvla_rl_pai hydra config): the runtime keeps events with
# t0 >= START_MARGIN and t0 + END_MARGIN <= CLIP_DURATION, then the loader
# (alpamayo_r1.load_physical_aiavdataset) asserts STRICTLY
# t0 > num_history_steps * time_step. The boundary case -- first kept event
# at exactly 1.6 s -- passes the runtime's >= filter and then crashes the
# loader ("t0_us must be greater than the history time range"; killed run
# alpamayo-rl-llm-judge-full-5ieeuh, 2026-07-22). We reproduce the runtime's
# event view here and drop clips whose FIRST kept event would trip the
# strict assert (the data packer always uses sample_index_in_clip=0).
_START_MARGIN_US = int(1.6 * 1_000_000)
_END_MARGIN_US = int(6.4 * 1_000_000)
_CLIP_DURATION_US = 20_000_000
_HISTORY_RANGE_US = int(16 * 0.1 * 1_000_000)  # num_history_steps * time_step


def _events_nonempty(events_cell: object) -> bool:
    """Mirror download_pai._ood_reasoning_events_nonempty: drop clips whose
    OOD events column is empty (they carry no judgeable reasoning)."""
    try:
        return events_cell is not None and len(events_cell) > 0  # type: ignore[arg-type]
    except TypeError:
        return False


def _first_kept_event_t0_us(events_cell: object) -> int | None:
    """First event t0 (µs) that survives the recipe runtime's margin filter,
    parsed exactly like pai_utils._read_reasoning_data (event_start_timestamp
    from a JSON string or an iterable of dicts). None when no event survives."""
    import json

    import numpy as np
    import pandas as pd

    if events_cell is None or (np.isscalar(events_cell) and pd.isna(events_cell)):
        return None
    parsed = json.loads(events_cell) if isinstance(events_cell, str) else events_cell
    if parsed is None or not hasattr(parsed, "__iter__"):
        return None
    for ev in parsed:
        if not (isinstance(ev, dict) and "event_start_timestamp" in ev):
            continue
        t0 = int(ev["event_start_timestamp"])
        if t0 >= _START_MARGIN_US and t0 + _END_MARGIN_US <= _CLIP_DURATION_US:
            return t0
    return None


def _loader_safe(events_cell: object) -> bool:
    """True when the clip's first runtime-kept event also satisfies the
    loader's STRICT history assert (t0 > history range)."""
    t0 = _first_kept_event_t0_us(events_cell)
    return t0 is not None and t0 > _HISTORY_RANGE_US


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = parser.parse_args()
    if args.num_chunks <= 0:
        parser.error("--num-chunks must be a positive integer")

    import pandas as pd
    from huggingface_hub import hf_hub_download

    clip_index = pd.read_parquet(
        hf_hub_download(args.repo_id, "clip_index.parquet", repo_type="dataset")
    )
    ood = pd.read_parquet(
        hf_hub_download(args.repo_id, "reasoning/ood_reasoning.parquet", repo_type="dataset")
    )
    if "events" in ood.columns:
        ood = ood[ood["events"].map(_events_nonempty)]
        n_before = len(ood)
        ood = ood[ood["events"].map(_loader_safe)]
        if len(ood) < n_before:
            print(
                f"[select_dense_ood_chunks] dropped {n_before - len(ood)} clip(s) whose first "
                "usable event t0 would fail the loader's strict history assert (t0 <= "
                f"{_HISTORY_RANGE_US} us) or has no event surviving the margin filter"
            )

    ood_ids = set(ood.index.astype(str))
    # Positional bool mask so .loc preserves clip_index's original index
    # dtype -- same convention as download_pai's mini-index writer.
    in_ood = clip_index.index.astype(str).isin(ood_ids)
    if not in_ood.any():
        raise SystemExit("No overlap between ood_reasoning and clip_index clip ids.")

    per_chunk = clip_index.loc[in_ood].groupby("chunk").size().sort_values(ascending=False)
    top_chunks = set(per_chunk.head(args.num_chunks).index)
    mini = clip_index.loc[in_ood & clip_index["chunk"].isin(top_chunks).to_numpy()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mini_path = args.output_dir / "clip_index_reasoning_mini.parquet"
    mini.to_parquet(mini_path)
    chunks = sorted(int(c) for c in top_chunks)
    chunks_path = args.output_dir / "dense_chunks.txt"
    chunks_path.write_text(" ".join(str(c) for c in chunks) + "\n")

    print(
        f"[select_dense_ood_chunks] selected {len(mini)} OOD clips across "
        f"{len(chunks)} densest chunks (of {per_chunk.size} OOD-bearing chunks, "
        f"{int(in_ood.sum())} OOD clips total); wrote {mini_path} and {chunks_path}"
    )


if __name__ == "__main__":
    main()
