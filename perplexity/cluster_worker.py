# Per-clip discrete-vs-diffusion worker. Runs INSIDE the self-bootstrapped
# Python 3.12 venv (training/bootstrap_venv.py) as a subprocess spawned by
# training/run.py's discrete_vs_diffusion_loop (which runs in Lilypad's base
# Python 3.10 env and cannot import alpamayo_r1 directly).
#
# For each assigned clip: get one real reasoning trace + a diffusion draw
# via the unmodified real entrypoint
# (sample_trajectories_from_data_with_vlm_rollout, return_extra=True, same
# call compare_discrete_continuous.py already validated locally), reuse that
# exact reasoning text for the discrete head
# (sample_discrete_action_tokens), draw NUM_DIFFUSION_SAMPLES more diffusion
# trajectories off the SAME frozen reasoning
# (sample_diffusion_trajectories_given_fixed_reasoning), then report how far
# the discrete trajectory sits from the diffusion cloud's centroid,
# normalized by the diffusion cloud's own pairwise self-spread (the "is
# discrete within the diffusion head's own noise floor" question this whole
# project is asking). One JSON summary line per clip goes to stdout under a
# fixed marker prefix (parseable via `lilypad workload logs --content-filter
# DISCRETE_VS_DIFFUSION_CLIP_SUMMARY`, same log-then-fetch pattern
# pref_pairs/fetch_from_logs.py established) since stdout flows through to
# the pod's own log stream regardless of which venv wrote it.
#
# A bounded number of representative top-down plots per cluster (not every
# clip -- 2 per cluster, PLOTS_PER_CLUSTER) get uploaded to S3 for visual
# spot-checking, same idea as plot_discrete_vs_diffusion.py's local plot but
# retrievable from a cluster run.

import argparse
import json
import logging
import sys

import boto3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

from cluster_data import extract_clip_to_dir
from dump_input_template import build_prompt
from s3_clip_loader import load_clip_from_s3_extract
from sample_diffusion_traj import sample_diffusion_trajectories_given_fixed_reasoning
from sample_discrete_traj import sample_discrete_action_tokens
from traj_tokenizer import detokenize_traj

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("discrete_vs_diffusion")

CHECKPOINT = "nvidia/Alpamayo-R1-10B"
MARKER = "DISCRETE_VS_DIFFUSION_CLIP_SUMMARY"
PLOT_MARKER = "DISCRETE_VS_DIFFUSION_PLOT"
NUM_DIFFUSION_SAMPLES = 10
PLOTS_PER_CLUSTER = 2
DIFFUSION_COLOR = "#2a78d6"  # dataviz categorical slot 1 (blue)
DISCRETE_COLOR = "#eb6834"  # dataviz categorical slot 8 (orange)


def ade_fde(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean/final xy displacement error between two (T, 3) trajectories."""
    xy_err = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)
    return float(xy_err.mean()), float(xy_err[-1])


def load_model() -> AlpamayoR1:
    # eager, not flash_attention_2 (the checkpoint's own default) -- no
    # flash-attn install on this first pass, see bootstrap_venv.py's module
    # docstring. NOT sdpa: transformers 4.57.1's init-time dispatch check
    # rejects it for this architecture ("AlpamayoR1 does not support an
    # attention implementation through torch.nn.functional.scaled_dot_
    # product_attention yet"), which killed canary7 -- reproduced and
    # eager-verified locally in a venv built by the same bootstrap_venv.py.
    return AlpamayoR1.from_pretrained(
        CHECKPOINT, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")


def process_clip(model, entry: dict, s3, bucket: str, plots_done: dict) -> dict:
    clip_id = entry["clip_id"]
    cluster = entry["event_cluster"]
    t0_us = entry["t0_us"]

    # Pull this clip's actual files (egomotion.parquet + 4 camera mp4s) out
    # of its S3 shard now -- the manifest only carries where to find them
    # (shard_key/group_start_offset), not pre-extracted local files.
    clip_dir = f"/tmp/clip_cache/{clip_id}"
    extract_clip_to_dir(entry["shard_key"], entry["group_start_offset"], clip_id, clip_dir)

    sample = load_clip_from_s3_extract(clip_dir, clip_id, t0_us=t0_us)

    # 1) Real reasoning + one diffusion draw via the unmodified demo entrypoint.
    prompt = build_prompt(model, sample)
    tokenized_data = {"input_ids": prompt["raw_input_ids"], **prompt["aux"]}
    model_inputs = {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": sample["ego_history_xyz"].to("cuda"),
        "ego_history_rot": sample["ego_history_rot"].to("cuda"),
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    cot_text = str(extra["cot"][0, 0, 0])
    meta_action_text = str(extra["meta_action"][0, 0, 0])
    reasoning_text = (
        f"{cot_text}<|cot_end|><|meta_action_start|>{meta_action_text}<|meta_action_end|>"
    )

    # 2) Discrete head, once, off that SAME reasoning.
    bin_ids = sample_discrete_action_tokens(model, sample, reasoning_text)
    hist_xyz, hist_rot = sample["ego_history_xyz"], sample["ego_history_rot"]
    xyz_discrete, _ = detokenize_traj(bin_ids.cpu(), hist_xyz, hist_rot)
    xyz_discrete = xyz_discrete[0, 0].cpu().float().numpy()  # (64, 3)

    # 3) Diffusion expert, K times, off that SAME reasoning.
    pred_xyz, _ = sample_diffusion_trajectories_given_fixed_reasoning(
        model, sample, reasoning_text, num_traj_samples=NUM_DIFFUSION_SAMPLES
    )
    xyz_diffusion = pred_xyz.cpu().float().numpy()  # (K, 64, 3)

    centroid = xyz_diffusion.mean(axis=0)
    mean_err, final_err = ade_fde(xyz_discrete, centroid)

    # Diffusion's own self-spread (mean pairwise ADE among the K samples) --
    # the noise floor normalized_score compares discrete's distance against.
    pairwise = [
        ade_fde(xyz_diffusion[i], xyz_diffusion[j])[0]
        for i in range(NUM_DIFFUSION_SAMPLES)
        for j in range(i + 1, NUM_DIFFUSION_SAMPLES)
    ]
    diffusion_self_spread = float(np.mean(pairwise)) if pairwise else 0.0
    normalized_score = (
        mean_err / diffusion_self_spread if diffusion_self_spread > 1e-6 else float("inf")
    )

    summary = {
        "clip_id": clip_id,
        "event_cluster": cluster,
        "t0_us": t0_us,
        "cot": cot_text,
        "mean_xy_err_to_centroid": mean_err,
        "final_xy_err_to_centroid": final_err,
        "diffusion_self_spread": diffusion_self_spread,
        "normalized_score": normalized_score,
    }
    logger.info("%s %s", MARKER, json.dumps(summary))

    # ALSO persist the summary to S3, one small JSON per clip next to the
    # plots. The stdout marker line above stays the quick-look convention,
    # but it is not durable: pod teardown can drop the final minutes of
    # stdout before the log shipper flushes (canary7's crash traceback and
    # canary8's summary lines were both lost this way). S3 is the source of
    # truth for collecting sweep results; logs are best-effort.
    s3.put_object(
        Bucket=bucket,
        Key=f"discrete_vs_diffusion_results/{cluster}/{clip_id}.json",
        Body=json.dumps(summary).encode("utf-8"),
    )

    if plots_done.get(cluster, 0) < PLOTS_PER_CLUSTER:
        fig, ax = plt.subplots(figsize=(7, 7))
        for k in range(NUM_DIFFUSION_SAMPLES):
            ax.plot(
                xyz_diffusion[k, :, 0],
                xyz_diffusion[k, :, 1],
                color=DIFFUSION_COLOR,
                alpha=0.5,
                linewidth=1.5,
                label=f"diffusion samples (K={NUM_DIFFUSION_SAMPLES})" if k == 0 else None,
            )
        ax.plot(
            xyz_discrete[:, 0],
            xyz_discrete[:, 1],
            color=DISCRETE_COLOR,
            linewidth=2.5,
            label="discrete head (detokenized)",
            zorder=5,
        )
        ax.scatter([0], [0], color="#0b0b0b", s=25, zorder=6, label="t0 (ego origin)")
        ax.set_xlabel("x (m, ego frame)")
        ax.set_ylabel("y (m, ego frame)")
        ax.set_title(f"{cluster}\n{clip_id}")
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        local_path = f"/tmp/{clip_id}_plot.png"
        fig.savefig(local_path, dpi=150)
        plt.close(fig)
        key = f"discrete_vs_diffusion_plots/{cluster}/{clip_id}.png"
        s3.upload_file(local_path, bucket, key)
        logger.info("%s %s", PLOT_MARKER, json.dumps({"clip_id": clip_id, "s3_key": key}))
        plots_done[cluster] = plots_done.get(cluster, 0) + 1

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--s3_bucket", default="research-datasets-chicago")
    args = parser.parse_args()

    with open(args.manifest) as f:
        entries = json.load(f)
    logger.info("cluster_worker: %d clips assigned to this rank", len(entries))

    model = load_model()
    s3 = boto3.client("s3")
    plots_done: dict[str, int] = {}

    n_ok, n_fail = 0, 0
    for entry in entries:
        try:
            process_clip(model, entry, s3, args.s3_bucket, plots_done)
            n_ok += 1
        except Exception:
            logger.exception("failed on clip %s", entry.get("clip_id"))
            n_fail += 1

    logger.info("cluster_worker done: %d ok, %d failed", n_ok, n_fail)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
