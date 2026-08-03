# SPDX-License-Identifier: Apache-2.0
"""
manifest.py -- builds the judge-dataset study's scene manifest: which
(clip_id, t0_us) scenes get 12 coupled rollouts generated and GPT-4o-scored.

Two arms, one output parquet (column `arm` distinguishes them):
  * ood:    every loader-safe OOD clip's FIRST kept event (the same clip set
            and decision moment rl_posttrain's training selector produces),
            all NVIDIA splits, with `split` and `event_cluster` carried
            through so the analysis can report per-category and keep val
            separate from train.
  * benign: `--benign-count` ordinary clips sampled (seeded) from
            clip_index.parquet's train split, excluding all OOD clip ids,
            with a fixed mid-clip t0 (8.0 s -- inside the 1.6 s history
            margin and the 6.4 s future horizon; no event annotation exists
            or is needed: the judge is GT-free). Controls for the
            "every training scene has a hazard" bias: measures hazard
            hallucination on empty roads and whether the judge rewards
            truthful nothing-unusual traces.

Event selection deliberately mirrors rl_posttrain/training/
select_dense_ood_chunks.py, NOT code_as_a_reward.ood_eval.manifest's
iter_ood_events: the ood_eval iterator keeps ALL sufficiently-late events
and ignores the end-of-clip margin, whereas training's data packer always
uses the FIRST event surviving BOTH margins and the loader then asserts
strictly t0 > 1.6 s on it (clips failing that are dropped entirely, not
advanced to their next event). The study must sample the same scenes
training would see.

Base-env-safe like code_as_a_reward/ood_eval/manifest.py (pandas +
huggingface_hub only); reuses its parquet fetch.

Usage:
  python -m judge_dataset.manifest --output judge_dataset/out/manifest.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from code_as_a_reward.ood_eval.manifest import REPO_ID, load_ood_reasoning_parquet

# Constants verbatim from select_dense_ood_chunks.py (which mirrors the
# recipe runtime's pai_utils margins and the loader's strict history assert).
START_MARGIN_US = 1_600_000
END_MARGIN_US = 6_400_000
CLIP_DURATION_US = 20_000_000
HISTORY_RANGE_US = 1_600_000

BENIGN_T0_US = 8_000_000  # mid-clip: > 1.6 s history margin, + 6.4 s <= 20 s


def first_kept_event(events_cell: object) -> dict | None:
    """The first event surviving the runtime margin filter, or None.
    Same walk as select_dense_ood_chunks._first_kept_event_t0_us, returning
    the whole event dict so the manifest can carry the CoC text."""
    if events_cell is None or (not hasattr(events_cell, "__iter__") and pd.isna(events_cell)):
        return None
    try:
        parsed = json.loads(events_cell) if isinstance(events_cell, str) else events_cell
    except (TypeError, ValueError):
        return None
    if parsed is None or not hasattr(parsed, "__iter__"):
        return None
    for ev in parsed:
        if not (isinstance(ev, dict) and "event_start_timestamp" in ev):
            continue
        t0 = int(ev["event_start_timestamp"])
        if t0 >= START_MARGIN_US and t0 + END_MARGIN_US <= CLIP_DURATION_US:
            return ev
    return None


def build_study_manifest(benign_count: int, seed: int) -> pd.DataFrame:
    ood_df = load_ood_reasoning_parquet()

    rows = []
    n_unsafe = 0
    for clip_id, row in ood_df.iterrows():
        ev = first_kept_event(row["events"])
        if ev is None:
            continue
        t0 = int(ev["event_start_timestamp"])
        if t0 <= HISTORY_RANGE_US:  # loader's strict assert drops the CLIP
            n_unsafe += 1
            continue
        rows.append(
            {
                "arm": "ood",
                "clip_id": str(clip_id),
                "t0_us": t0,
                "split": str(row["split"]),
                "event_cluster": str(row["event_cluster"]),
                "gt_coc": ev.get("coc"),
            }
        )
    print(
        f"[judge_dataset.manifest] ood arm: {len(rows)} loader-safe clips "
        f"({n_unsafe} dropped on the strict t0 assert)"
    )

    if benign_count > 0:
        from huggingface_hub import hf_hub_download

        clip_index = pd.read_parquet(
            hf_hub_download(REPO_ID, "clip_index.parquet", repo_type="dataset")
        )
        ood_ids = set(ood_df.index.astype(str))
        pool = clip_index[
            (clip_index["split"] == "train")
            & ~clip_index.index.astype(str).isin(ood_ids)
        ]
        benign = pool.sample(n=benign_count, random_state=seed)
        rows += [
            {
                "arm": "benign",
                "clip_id": str(cid),
                "t0_us": BENIGN_T0_US,
                "split": "train",
                "event_cluster": None,
                "gt_coc": None,
            }
            for cid in benign.index
        ]

    df = pd.DataFrame(rows)
    assert df["clip_id"].is_unique, "manifest must have one scene per clip"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benign-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    df = build_study_manifest(args.benign_count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    print(
        f"[judge_dataset.manifest] {len(df)} scenes -> {args.output}\n"
        f"  by arm: {df['arm'].value_counts().to_dict()}\n"
        f"  ood by split: {df[df.arm == 'ood']['split'].value_counts().to_dict()}"
    )


if __name__ == "__main__":
    main()
