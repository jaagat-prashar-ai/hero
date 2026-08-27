# SPDX-License-Identifier: Apache-2.0
"""Track 1 GPU worker: generate 6 CoC rollouts per test event for ONE arm.

Runs inside the bootstrapped Python 3.12 Alpamayo venv (see
code_as_a_reward/ood_eval/bootstrap_venv.py) as a subprocess of
reasoning/run_track1_submissions.py. One arm == one GPU (Ray sets
CUDA_VISIBLE_DEVICES for the parent rank; we inherit it).

Model loading: always start from the released nvidia/Alpamayo-1.5-10B (the
base every RL run in this repo post-trained), then -- for trained arms --
overwrite the VLM weights from a convert_cosmos_rl_checkpoint.py export dir.
The export contains only the VLM backbone (recipes SKILL.md), which is
exactly the RL-trained pathway: CoC text AND the autoregressive trajectory
tokens both come from the VLM; the (untouched) base action expert is unused
by sample_rollout_group. Loading is strict about unexpected keys and about
a minimum match fraction so a silent name mismatch cannot masquerade as a
"trained" run of base weights.

Output: appends one JSON line per event to --output-jsonl
({"submission_key", "rollouts": [6 CoC strings], "waypoints", ...}), with
skip-if-done resume on restart.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

MIN_OVERLAY_MATCH_FRACTION = 0.95


def _load_done_keys(output_jsonl: str) -> set[str]:
    """Keys with REAL rollouts only -- errored events must be retried on
    resume, not frozen as permanently-empty submission keys."""
    done: set[str] = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("rollouts"):
                        done.add(rec["submission_key"])
                except Exception:
                    logger.warning("skipping unparseable resume line: %r", line[:100])
    return done


def _overlay_export_weights(model, export_dir: str) -> None:
    """Overwrite matching model weights from a cosmos-rl export directory."""
    import torch
    from safetensors.torch import load_file

    index_path = os.path.join(export_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    state: dict[str, torch.Tensor] = {}
    for name in shard_names:
        state.update(load_file(os.path.join(export_dir, name)))
    logger.info("export %s: %d tensors in %d shards", export_dir, len(state), len(shard_names))

    model_keys = set(model.state_dict().keys())
    matched = set(state) & model_keys
    unexpected = sorted(set(state) - model_keys)
    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)}/{len(state)} export keys not in the base model "
            f"(sample: {unexpected[:5]}) -- key convention changed, refusing to "
            "generate with silently-unloaded weights"
        )
    frac = len(matched) / max(len(state), 1)
    if frac < MIN_OVERLAY_MATCH_FRACTION:
        raise RuntimeError(f"only {frac:.1%} of export keys matched the base model")

    # Sanity: prove the overlay actually changes weights.
    probe_key = next(iter(sorted(matched)))
    before = model.state_dict()[probe_key].float().norm().item()

    missing, unexpected2 = model.load_state_dict(state, strict=False)
    assert not unexpected2, unexpected2[:5]
    after = model.state_dict()[probe_key].float().norm().item()
    logger.info(
        "overlaid %d/%d tensors (%.1f%%); %d base tensors untouched (non-VLM); "
        "probe %s norm %.4f -> %.4f",
        len(matched), len(state), 100 * frac, len(missing), probe_key, before, after,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--export-dir", default=None,
                        help="cosmos-rl HF export dir; omit for the base SFT model")
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--fill-dir", default=None,
                        help="egomotion sidecar dir for clips whose shard member is missing")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--max-events", type=int, default=None,
                        help="smoke-test truncation; None = all events")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    import torch

    from code_as_a_reward.clipgen.rollout_sampler import load_model, sample_rollout_group
    from reasoning.track1_data import iter_track1_samples

    torch.manual_seed(args.seed)

    shard_paths = sorted(
        os.path.join(args.shards_dir, n)
        for n in os.listdir(args.shards_dir)
        if n.endswith(".tar")
    )
    if not shard_paths:
        raise RuntimeError(f"no .tar shards under {args.shards_dir}")
    logger.info("[%s] %d shards, group_size=%d", args.arm, len(shard_paths), args.group_size)

    done = _load_done_keys(args.output_jsonl)
    logger.info("[%s] resume: %d events already done", args.arm, len(done))
    iter_kwargs = {"fill_dir": args.fill_dir} if args.fill_dir else {}

    model = load_model()
    if args.export_dir:
        _overlay_export_weights(model, args.export_dir)
    else:
        logger.info("[%s] base SFT weights (no export overlay)", args.arm)
    model.eval()

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    n_done, n_err = 0, 0
    t_start = time.time()
    with open(args.output_jsonl, "a") as out:
        for sample in iter_track1_samples(shard_paths, **iter_kwargs):
            key = sample["submission_key"]
            if key in done:
                continue
            if args.max_events is not None and n_done >= args.max_events:
                logger.info("[%s] --max-events %d reached, stopping", args.arm, args.max_events)
                break

            if "error" in sample:
                record = {"submission_key": key, "clip_id": sample["clip_id"],
                          "event_idx": sample["event_idx"], "error": sample["error"]}
                n_err += 1
            else:
                try:
                    with torch.no_grad():
                        rollouts = sample_rollout_group(model, sample, n=args.group_size)
                    record = {
                        "submission_key": key,
                        "clip_id": sample["clip_id"],
                        "event_idx": sample["event_idx"],
                        "t0_us": sample["t0_us"],
                        "t0_us_used": sample["t0_us_used"],
                        "clamped": sample["clamped"],
                        "rollouts": [r["coc_text"] for r in rollouts],
                        "waypoints": [r["waypoints"] for r in rollouts],
                    }
                except Exception as exc:
                    logger.exception("[%s] event %s failed", args.arm, key)
                    record = {"submission_key": key, "clip_id": sample["clip_id"],
                              "event_idx": sample["event_idx"], "error": str(exc)}
                    n_err += 1

            out.write(json.dumps(record) + "\n")
            out.flush()
            n_done += 1
            if n_done % 10 == 0:
                rate = n_done / max(time.time() - t_start, 1e-6)
                logger.info("[%s] %d events done (%d errors), %.1f events/min",
                            args.arm, n_done, n_err, 60 * rate)

    logger.info("[%s] finished: %d new events, %d errors", args.arm, n_done, n_err)
    if n_err and n_err >= max(n_done // 4, 5):
        logger.error("[%s] high error rate", args.arm)
        sys.exit(2)


if __name__ == "__main__":
    main()
