"""Judge-based faithfulness-vs-accuracy analysis over judgments_full.jsonl.

x-axis: gpt-5-6-terra causal-faithfulness score (0-100) per rollout, where the
judge saw the reasoning text plus the raw 64 waypoints and a kinematics
summary. Averaged per event for the scatter.
y-axis: ADE against GT egomotion (same as analyze_offline_full.py).

Writes misalignment_figure_judge.png and prints per-event r, bootstrap CI,
and the within-event contrast.
"""
import glob
import importlib.util
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.1

spec = importlib.util.spec_from_file_location(
    "mfig", os.path.join(HERE, "make_misalignment_figure.py"))
mfig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mfig)


def main():
    rows = [r for r in map(json.loads, open(os.path.join(HERE, "judgments_full.jsonl")))
            if "score" in r]
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["arm", "key", "ridx"], keep="last")
    print(f"judged rollouts: {len(df)}, arms: {sorted(df['arm'].unique())}")
    print("score distribution:", np.percentile(df["score"], [5, 25, 50, 75, 95]))

    preds, t0s = {}, {}
    for arm in df["arm"].unique():
        with open(os.path.join(HERE, f"rollouts_{arm}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r and r.get("rollouts") and all(r["rollouts"]):
                    preds[(arm, r["submission_key"])] = r["waypoints"]
                    t0s[r["submission_key"]] = r["t0_us_used"]

    ego_paths = {os.path.basename(p).split(".")[0]: p
                 for p in glob.glob(os.path.join(HERE, "test_egomotion", "*.egomotion.parquet"))}
    gt_cache = {}

    def gt_for(key):
        if key not in gt_cache:
            gt_cache[key] = mfig.ego_frame_waypoints(
                pd.read_parquet(ego_paths[key.rsplit("_", 1)[0]]), t0s[key])
        return gt_cache[key]

    ades = []
    for _, row in df.iterrows():
        gt = gt_for(row["key"])
        pred = np.array(preds[(row["arm"], row["key"])][row["ridx"]])
        ades.append(float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1).mean()))
    df["ade"] = ades

    ev = df.groupby("key").agg(score=("score", "mean"), ade=("ade", "mean"),
                               n=("score", "size")).reset_index()
    x, y = ev["score"].to_numpy(), ev["ade"].to_numpy()
    r = np.corrcoef(x, y)[0, 1]
    z, se = np.arctanh(r), 1 / np.sqrt(len(ev) - 3)
    print(f"events: {len(ev)}; per-event r = {r:.3f}, "
          f"95% CI [{np.tanh(z-1.96*se):.2f}, {np.tanh(z+1.96*se):.2f}]")

    # within-event contrast, scenario difficulty removed
    df["ade_dm"] = df["ade"] - df.groupby("key")["ade"].transform("mean")
    per_rollout_r = np.corrcoef(df["score"], df["ade_dm"])[0, 1]
    print(f"per-rollout within-event r = {per_rollout_r:.3f} (n={len(df)})")
    hi = df[df["score"] >= 70]["ade_dm"]
    lo = df[df["score"] <= 24]["ade_dm"]
    print(f"within-event demeaned ADE: entailed(>=70) n={len(hi)} {hi.mean():+.3f} m "
          f"vs contradicted(<=24) n={len(lo)} {lo.mean():+.3f} m "
          f"(diff {hi.mean()-lo.mean():+.3f})")

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(x, y, s=22, color="#3477b4", alpha=0.7, zorder=3)
    xs = np.linspace(x.min() - 2, x.max() + 2, 100)
    b1, b0 = np.polyfit(x, y, 1)
    rng = np.random.default_rng(0)
    boots = np.empty((1000, len(xs)))
    for i in range(1000):
        idx = rng.integers(0, len(x), len(x))
        bb1, bb0 = np.polyfit(x[idx], y[idx], 1)
        boots[i] = bb1 * xs + bb0
    lo_b, hi_b = np.percentile(boots, [2.5, 97.5], axis=0)
    ax.plot(xs, b1 * xs + b0, color="#3477b4", lw=2, zorder=2)
    ax.fill_between(xs, lo_b, hi_b, color="#3477b4", alpha=0.18, zorder=1)
    ax.set_xlabel("Faithfulness (gpt-5-6-terra judge on raw trajectory, event mean)",
                  fontsize=11)
    ax.set_ylabel("Decision Error (ADE vs GT, m)", fontsize=11)
    ax.set_title(f"r = {r:.2f}, n = {len(ev)} events, {len(df)} rollouts", fontsize=11)
    ax.grid(alpha=0.3, ls="--")
    fig.tight_layout()
    out = os.path.join(HERE, "misalignment_figure_judge.png")
    fig.savefig(out, dpi=180)
    print("wrote", out)


if __name__ == "__main__":
    main()
