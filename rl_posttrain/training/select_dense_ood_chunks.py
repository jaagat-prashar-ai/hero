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


def _events_nonempty(events_cell: object) -> bool:
    """Mirror download_pai._ood_reasoning_events_nonempty: drop clips whose
    OOD events column is empty (they carry no judgeable reasoning)."""
    try:
        return events_cell is not None and len(events_cell) > 0  # type: ignore[arg-type]
    except TypeError:
        return False


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
