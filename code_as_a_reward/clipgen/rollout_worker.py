# SPDX-License-Identifier: Apache-2.0
"""GPU worker: samples a real Alpamayo rollout group per target clip via
rollout_sampler.py, writes one <clip_id>.json per clip. Runs INSIDE the
bootstrapped alpamayo1_5 Python 3.12 venv (ood_eval.bootstrap_venv), invoked
as a subprocess by run_real_rollout_gen.py -- same process-boundary split as
code_as_a_reward/ood_eval/run.py + worker.py, and for the same reason: torch/
alpamayo1_5 aren't importable in Lilypad's base Python 3.10 env.

No LLM calls here (keeps Anthropic/OpenAI SDKs out of this venv); the
base-env driver runs the LLM generation+gate loop afterward, reading these
files (see run_prototype.py's module docstring).

Resumable: skips clip_ids whose output file already exists, so a preempted
node (or a re-run after run_real_rollout_gen.py restores prior progress
from S3) doesn't resample clips it already has.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets_json", required=True, help='JSON list of {"clip_id","t0_us"}')
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--group_size", type=int, default=12)
    args = ap.parse_args()

    from code_as_a_reward.clipgen.rollout_sampler import load_model, sample_rollout_group_for_clip

    with open(args.targets_json) as f:
        targets = json.load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("loading model ...")
    model = load_model()
    logger.info("model loaded")

    n_ok = n_skipped = n_failed = 0
    for t in targets:
        clip_id, t0_us = t["clip_id"], t["t0_us"]
        out_path = os.path.join(args.output_dir, f"{clip_id}.json")
        if os.path.exists(out_path):
            n_skipped += 1
            continue
        try:
            rollouts = sample_rollout_group_for_clip(model, clip_id, t0_us, n=args.group_size)
            with open(out_path, "w") as f:
                json.dump({"clip_id": clip_id, "t0_us": t0_us, "rollouts": rollouts}, f)
            logger.info("clip %s: sampled %d rollouts", clip_id, len(rollouts))
            n_ok += 1
        except Exception:
            logger.exception("clip %s: rollout sampling failed, skipping", clip_id)
            n_failed += 1

    logger.info("done: %d ok, %d failed, %d already-done skipped", n_ok, n_failed, n_skipped)


if __name__ == "__main__":
    main()
