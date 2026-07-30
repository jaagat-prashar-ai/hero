# SPDX-License-Identifier: Apache-2.0
"""
analyze.py — loads worker.py's output JSONL and produces:

  * a flat, one-row-per-event summary (pandas-friendly parquet/CSV) with
    both branches' TraceReward scalars + per-family PASS/FAIL/ABSTAIN
    counts, prefixed gt_/model_;
  * an aggregate table grouped by event_cluster x branch: mean reward,
    atomic_precision, decided_fraction, and total PASS/FAIL/ABSTAIN counts
    per claim family.

This is the "successful points / edge cases / failure cases" view: high,
stable atomic_precision with low n_fail is a successful pattern; a family
with a high abstain share is an edge case (verifier can't decide, not that
it decided wrong); a family with a high, consistent fail share is a
candidate systematic failure (verifier bug, or the model/dataset genuinely
being unfaithful there -- the gt_ vs model_ split is what tells those apart).

Base-env-safe (pandas only, no physical_ai_av/torch) -- can run against a
results file synced from S3 without needing the bootstrapped venv at all.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_FAMILIES = ("commitment", "perceptual", "causal")


def _branch_summary(branch: dict | None) -> dict:
    if branch is None:
        return {"available": False}
    reward = branch["reward"]
    out = {
        "available": True,
        "scene_available": branch["scene_available"],
        "reward": reward["reward"],
        "atomic_precision": reward["atomic_precision"],
        "causal_precision": reward["causal_precision"],
        "decided_fraction": reward["decided_fraction"],
        "unparsed_char_fraction": reward["unparsed_char_fraction"],
    }
    for family in _FAMILIES:
        out[f"n_pass_{family}"] = reward["n_pass"].get(family, 0)
        out[f"n_fail_{family}"] = reward["n_fail"].get(family, 0)
        out[f"n_abstain_{family}"] = reward["n_abstain"].get(family, 0)
    return out


def load_results(path: str) -> pd.DataFrame:
    """One row per event, flattened. `has_model` distinguishes events scored
    on ground truth only (model branch skipped for that run) from events
    with a genuine gt-vs-model comparison."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            flat = {
                "clip_id": row["clip_id"],
                "t0_us": row["t0_us"],
                "scene_id": row["scene_id"],
                "event_cluster": row["event_cluster"],
                "has_model": row.get("model") is not None,
            }
            flat.update({f"gt_{k}": v for k, v in _branch_summary(row["ground_truth"]).items()})
            flat.update({f"model_{k}": v for k, v in _branch_summary(row.get("model")).items()})
            rows.append(flat)
    df = pd.DataFrame(rows)
    logger.info("loaded %d events from %s (%d with a model rollout)", len(df), path, df["has_model"].sum() if len(df) else 0)
    return df


def summarize_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """event_cluster x {mean reward/precision/decided_fraction, total
    PASS/FAIL/ABSTAIN per family} for both branches."""
    if df.empty:
        return df

    scalar_cols = [
        c
        for c in df.columns
        if c.startswith(("gt_", "model_"))
        and c.split("_", 1)[1] in ("reward", "atomic_precision", "causal_precision", "decided_fraction")
    ]
    agg = df.groupby("event_cluster")[scalar_cols].mean()
    agg.insert(0, "n_events", df.groupby("event_cluster").size())

    count_cols = [c for c in df.columns if "_n_pass_" in c or "_n_fail_" in c or "_n_abstain_" in c]
    for col in count_cols:
        agg[col] = df.groupby("event_cluster")[col].sum()

    return agg.reset_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_jsonl")
    ap.add_argument("--out_flat_parquet", default=None, help="write the flat one-row-per-event table here")
    ap.add_argument("--out_summary_csv", default=None, help="write the event_cluster aggregate table here")
    args = ap.parse_args()

    df = load_results(args.results_jsonl)
    summary = summarize_by_cluster(df)

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary.to_string(index=False))

    if args.out_flat_parquet:
        df.to_parquet(args.out_flat_parquet, index=False)
        logger.info("wrote flat table to %s", args.out_flat_parquet)
    if args.out_summary_csv:
        summary.to_csv(args.out_summary_csv, index=False)
        logger.info("wrote summary to %s", args.out_summary_csv)


if __name__ == "__main__":
    main()
