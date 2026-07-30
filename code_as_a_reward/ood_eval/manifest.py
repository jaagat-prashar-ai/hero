# SPDX-License-Identifier: Apache-2.0
"""
manifest.py — builds the list of (clip_id, t0_us, ground-truth CoC) events
to run the commitment/perceptual verifiers over: every event across all
1740 clips in nvidia/PhysicalAI-Autonomous-Vehicles' reasoning/ood_reasoning
.parquet (9 event_cluster types).

Deliberately base-env-safe: only pandas + huggingface_hub, no physical_ai_av
(that needs Python >= 3.11; see obstacle_tracks.py's module docstring) and no
torch/alpamayo1_5 -- so run.py (Lilypad's base Python 3.10 worker) can build
the full manifest before bootstrapping the venv that runs worker.py.

Row/field shapes taken directly from perplexity/sample_ood_clips.py, which
already reads this exact file (ood_df.loc[clip_id]["events"], JSON list of
{event_start_timestamp, coc}) -- reused rather than re-derived.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
OOD_REASONING_FILENAME = "reasoning/ood_reasoning.parquet"

# load_physical_aiavdataset asserts t0_us > num_history_steps * time_step *
# 1e6 (16 * 0.1 * 1e6 = 1.6e6); this small buffer avoids landing exactly on
# that boundary. Same value perplexity/sample_ood_clips.py uses.
MIN_T0_US = 1_700_000


@dataclasses.dataclass
class OODEvent:
    """One (clip, reasoning moment) to verify. `rank_in_clip` is the 0-indexed
    position within this clip's own events list -- some clips have more than
    one event; this is NOT a global ordering."""

    clip_id: str
    t0_us: int
    gt_coc: str
    event_cluster: str
    rank_in_clip: int

    def scene_id(self) -> str:
        return f"{self.clip_id}_{self.t0_us}"


def load_ood_reasoning_parquet(local_path: str | None = None) -> pd.DataFrame:
    """Load reasoning/ood_reasoning.parquet, indexed by clip_id.

    Downloads via HF (requires an HF token authorized for the gated dataset)
    unless `local_path` is given -- same fetch perplexity/sample_ood_clips.py
    already uses, just without physical_ai_av's own dataset interface (this
    file needs nothing beyond the raw parquet)."""
    if local_path is not None:
        return pd.read_parquet(local_path)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=OOD_REASONING_FILENAME)
    return pd.read_parquet(path)


def iter_ood_events(df: pd.DataFrame, *, min_t0_us: int = MIN_T0_US):
    """Yield every valid OODEvent across every clip's events list.

    ~9/1740 clips have a null `events` cell (known upstream gap, confirmed by
    perplexity/sample_ood_clips.py) -- skipped, not an error. An event whose
    t0_us doesn't clear `min_t0_us` is skipped too (not enough history for
    load_physical_aiavdataset's t0 assertion) and counted separately so the
    caller can see how much of the corpus that filter actually drops.
    """
    n_clips_no_events = 0
    n_events_seen = 0
    n_events_too_early = 0
    n_events_yielded = 0
    for clip_id, row in df.iterrows():
        if pd.isna(row["events"]):
            n_clips_no_events += 1
            continue
        events = json.loads(row["events"]) if isinstance(row["events"], str) else row["events"]
        for i, e in enumerate(events):
            n_events_seen += 1
            if e["event_start_timestamp"] <= min_t0_us:
                n_events_too_early += 1
                continue
            n_events_yielded += 1
            yield OODEvent(
                clip_id=str(clip_id),
                t0_us=int(e["event_start_timestamp"]),
                gt_coc=e["coc"],
                event_cluster=str(row.get("event_cluster")),
                rank_in_clip=i,
            )
    logger.info(
        "ood_reasoning.parquet: %d clips with no events, %d/%d events kept "
        "(%d dropped as too-early for t0)",
        n_clips_no_events, n_events_yielded, n_events_seen, n_events_too_early,
    )


def build_manifest(
    local_path: str | None = None,
    *,
    min_t0_us: int = MIN_T0_US,
    max_events: int | None = None,
    clip_ids: list[str] | None = None,
) -> list[OODEvent]:
    """Full manifest as a list (not a generator) so it can be JSON-dumped for
    a subprocess worker to read. `clip_ids`, if given, restricts to those
    clips (smoke-test subsetting); `max_events` truncates the final list
    (smoke-test sizing) -- applied in that order.
    """
    df = load_ood_reasoning_parquet(local_path)
    if clip_ids is not None:
        wanted = set(clip_ids)
        df = df[df.index.astype(str).isin(wanted)]
    events = list(iter_ood_events(df, min_t0_us=min_t0_us))
    if max_events is not None:
        events = events[:max_events]
    return events


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local_path", default=None, help="local ood_reasoning.parquet, skips HF fetch")
    ap.add_argument("--max_events", type=int, default=None)
    ap.add_argument("--out_json", default=None, help="write the manifest (list of OODEvent) here")
    args = ap.parse_args()

    events = build_manifest(args.local_path, max_events=args.max_events)
    logger.info("built manifest: %d events", len(events))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump([dataclasses.asdict(e) for e in events], f)
        logger.info("wrote %s", args.out_json)


if __name__ == "__main__":
    main()
