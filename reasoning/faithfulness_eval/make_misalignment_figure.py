"""Two-panel reasoning-decision misalignment figure from the 2026-08-27
gpt-4o faithfulness eval (judgments.jsonl) + fleet1 rollouts + GT egomotion.

Panel (a): per-event mean judge score (8 judgments: 4 arms x 2 rollouts) vs
mean ADE of those same rollouts against GT egomotion, with OLS fit +
bootstrap CI band.
Panel (b): case study — reasoning says yield/stop, trajectory accelerates.

Run from repo root:  python3 reasoning/faithfulness_eval/make_misalignment_figure.py
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EGO_DIR = os.path.join(HERE, "test_egomotion")
ARMS = ["base_sft", "consistency", "llm_judge", "code_v2_main"]
DT = 0.1
N_WP = 64

# Panel (b) case: "Stop due to pedestrians in the crosswalk" while the model
# accelerates from standstill into it (GT stays stopped for the full 6.4 s).
CASE = {"arm": "code_v2_main", "key": "34055344-495d-41f3-91b2-2d143de88d20_1", "ridx": 1}


def ego_frame_waypoints(df, t0_us):
    """Future 6.4 s of GT motion at 10 Hz in the ego frame at t0 (x fwd, y left)."""
    ts = df["timestamp_us"].to_numpy()
    pos = df[["x", "y", "z"]].to_numpy()
    i0 = int(np.argmin(np.abs(ts - t0_us)))
    p0 = pos[i0]
    qx, qy, qz, qw = df[["qx", "qy", "qz", "qw"]].to_numpy()[i0]
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    tq = t0_us + np.arange(1, N_WP + 1) * DT * 1e6
    j = np.searchsorted(ts, tq)
    if j[-1] >= len(ts):
        return None
    j = np.maximum(j, 1)
    a = ((tq - ts[j - 1]) / (ts[j] - ts[j - 1]))[:, None]
    p = (1 - a) * pos[j - 1] + a * pos[j]
    return (p - p0) @ R  # row-vectors: world -> ego via R (== R.T applied per point)


def main():
    judgments = [json.loads(l) for l in open(os.path.join(HERE, "judgments.jsonl"))]
    rollouts = {}
    for arm in ARMS:
        rollouts[arm] = {
            r["submission_key"]: r
            for r in map(json.loads, open(os.path.join(HERE, f"rollouts_{arm}.jsonl")))
        }
    ego_paths = {
        os.path.basename(p).split(".")[0]: p
        for p in glob.glob(os.path.join(EGO_DIR, "*.egomotion.parquet"))
    }

    gt_cache = {}

    def gt_for(key):
        if key in gt_cache:
            return gt_cache[key]
        clip = key.rsplit("_", 1)[0]
        wp = None
        if clip in ego_paths:
            t0 = rollouts[ARMS[0]][key]["t0_us_used"]
            wp = ego_frame_waypoints(pd.read_parquet(ego_paths[clip]), t0)
        gt_cache[key] = wp
        return wp

    # Per-rollout ADE joined to judge score.
    rows = []
    for j in judgments:
        gt = gt_for(j["key"])
        if gt is None:
            continue
        pred = np.array(rollouts[j["arm"]][j["key"]]["waypoints"][j["ridx"]])
        ade = float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1).mean())
        rows.append({**j, "ade": ade})
    df = pd.DataFrame(rows)
    ev = df.groupby("key").agg(score=("score", "mean"), ade=("ade", "mean"),
                               n=("score", "size")).reset_index()
    print(f"rollouts with GT: {len(df)}/800; events: {len(ev)}")
    r = np.corrcoef(ev["score"], ev["ade"])[0, 1]
    print(f"per-event pearson r = {r:.3f}")

    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28,
                          left=0.06, right=0.97, top=0.93, bottom=0.19)

    # ---- Panel (a): scatter + OLS fit + bootstrap CI band ----
    ax = fig.add_subplot(gs[0])
    x, y = ev["score"].to_numpy(), ev["ade"].to_numpy()
    ax.scatter(x, y, s=28, color="#3477b4", alpha=0.85, zorder=3)
    xs = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
    b1, b0 = np.polyfit(x, y, 1)
    rng = np.random.default_rng(0)
    boots = np.empty((1000, len(xs)))
    for i in range(1000):
        idx = rng.integers(0, len(x), len(x))
        bb1, bb0 = np.polyfit(x[idx], y[idx], 1)
        boots[i] = bb1 * xs + bb0
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    ax.plot(xs, b1 * xs + b0, color="#3477b4", lw=2, zorder=2)
    ax.fill_between(xs, lo, hi, color="#3477b4", alpha=0.18, zorder=1)
    case_ev = ev[ev["key"] == CASE["key"]]
    if len(case_ev):
        ax.scatter(case_ev["score"], case_ev["ade"], marker="^", s=130,
                   facecolors="none", edgecolors="red", linewidths=1.8, zorder=4)
    ax.set_xlabel("Faithfulness Score (gpt-4o judge, event mean)", fontsize=11)
    ax.set_ylabel("Decision Error (ADE vs GT, m)", fontsize=11)
    ax.grid(alpha=0.3, ls="--")
    fig.text(0.29, 0.03, f"(a) Reasoning–Decision Correlation  (r = {r:.2f}, "
             f"n = {len(ev)} events)", ha="center", fontsize=11.5)

    # ---- Panel (b): case study ----
    axb = fig.add_subplot(gs[1])
    axb.set_axis_off()
    case_r = rollouts[CASE["arm"]][CASE["key"]]
    text = case_r["rollouts"][CASE["ridx"]]
    pred = np.array(case_r["waypoints"][CASE["ridx"]])
    gt = gt_for(CASE["key"])
    v = np.linalg.norm(np.diff(pred[:, :2], axis=0), axis=1) / DT
    jrec = next(j for j in judgments if (j["arm"], j["key"], j["ridx"]) ==
                (CASE["arm"], CASE["key"], CASE["ridx"]))

    import textwrap

    axb.text(0.5, 1.0, "Right Reasoning", ha="center", va="top", fontsize=13,
             weight="bold", transform=axb.transAxes)
    box_green = dict(boxstyle="round,pad=0.55", fc="#dcefd8", ec="none")
    axb.text(0.5, 0.87, textwrap.fill(f"Reasoning: “{text}”", 58),
             ha="center", va="center", fontsize=10.5, style="italic",
             transform=axb.transAxes, bbox=box_green)
    axb.text(0.5, 0.70,
             textwrap.fill(f"Judge: expected “{jrec['expected_long']}” — "
                           f"{jrec['why']}", 62),
             ha="center", va="center", fontsize=9.5,
             transform=axb.transAxes, bbox=box_green)
    axb.annotate("", xy=(0.5, 0.50), xytext=(0.5, 0.60),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    axb.text(0.5, 0.47, "Wrong Decision", ha="center", va="top", fontsize=13,
             weight="bold", transform=axb.transAxes)
    axb.text(0.5, 0.35,
             textwrap.fill(f"Executed trajectory: accelerates "
                           f"{v[:5].mean():.1f} → {v[-5:].mean():.1f} m/s, "
                           f"advancing {v.sum() * DT:.1f} m into the crosswalk "
                           f"(GT remains stopped)", 62),
             ha="center", va="center", fontsize=10, color="#8b1a1a",
             transform=axb.transAxes,
             bbox=dict(boxstyle="round,pad=0.55", fc="#f6d7d4", ec="none"))
    # Speed-profile inset: model vs GT over the 6.4 s horizon.
    axi = axb.inset_axes([0.22, 0.0, 0.56, 0.24])
    t = np.arange(1, N_WP) * DT
    axi.plot(t, np.linalg.norm(np.diff(pred[:, :2], axis=0), axis=1) / DT,
             color="#c0392b", lw=2, label="model")
    if gt is not None:
        axi.plot(t, np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1) / DT,
                 color="#2e7d32", lw=2, ls="--", label="GT (stays stopped)")
    axi.set_xlabel("time (s)", fontsize=8)
    axi.set_ylabel("speed (m/s)", fontsize=8)
    axi.tick_params(labelsize=7)
    axi.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.text(0.75, 0.03, "(b) Misalignment", ha="center", fontsize=11.5)

    out = os.path.join(HERE, "misalignment_figure.png")
    fig.savefig(out, dpi=180)
    print("wrote", out)


if __name__ == "__main__":
    main()
