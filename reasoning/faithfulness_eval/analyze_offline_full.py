"""Full-sample OFFLINE faithfulness-vs-accuracy analysis (no API).

x-axis: the GRPO consistency-reward parser (coc_consistency_reward) scored on
all local rollouts — per-rollout axis-level consistency in {0, 0.5, 1}
(unparseable excluded), averaged per event for the scatter.
y-axis: ADE against GT egomotion (same as make_misalignment_figure).

Writes misalignment_figure_full.png (panel (a) full-scale + panel (b) case
study unchanged) and prints the within-event contrast.
"""
import glob
import importlib.util
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DT = 0.1

spec = importlib.util.spec_from_file_location(
    "mfig", os.path.join(HERE, "make_misalignment_figure.py"))
mfig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mfig)

ccr_path = os.path.join(REPO, "third_party", "alpamayo-recipes", "recipes",
                        "alpamayo1_x_rl", "rewards", "coc_consistency_reward.py")
ccr_spec = importlib.util.spec_from_file_location("_ccr", ccr_path)
ccr = importlib.util.module_from_spec(ccr_spec)
sys.modules["_ccr"] = ccr
ccr_spec.loader.exec_module(ccr)


def controls_from_waypoints(wp):
    """(accel, kappa, v0) derived from 64 ego-frame waypoints @ 10 Hz."""
    w = np.asarray(wp, dtype=float)[:, :2]
    d = np.diff(w, axis=0)
    v = np.linalg.norm(d, axis=1) / DT  # 63
    accel = np.diff(v, prepend=v[0]) / DT  # 63
    heading = np.arctan2(d[:, 1], d[:, 0])
    dh = np.diff(np.unwrap(heading), prepend=0.0)  # 63
    ds = np.maximum(v * DT, 1e-3)
    kappa = dh / ds
    kappa[v < 0.3] = 0.0  # heading is noise when nearly stopped
    return accel, kappa, float(v[:5].mean())


def axis_score(diag):
    """Per-rollout faithfulness in {0, 0.5, 1}: fraction of axes consistent."""
    return (float(diag["lon_consistent"]) + float(diag["lat_consistent"])) / 2.0


def main():
    rows = []
    for arm in mfig.ARMS:
        with open(os.path.join(HERE, f"rollouts_{arm}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if "error" in r or not r.get("rollouts") or not all(r["rollouts"]):
                    continue
                for ridx, (coc, wp) in enumerate(zip(r["rollouts"], r["waypoints"])):
                    accel, kappa, v0 = controls_from_waypoints(wp)
                    _, diag = ccr.score_consistency(coc, accel, kappa, v0)
                    rows.append({
                        "arm": arm, "key": r["submission_key"], "ridx": ridx,
                        "t0": r["t0_us_used"], "unparseable": diag["unparseable"],
                        "faith": None if diag["unparseable"] else axis_score(diag),
                    })
    df = pd.DataFrame(rows)
    print(f"rollouts: {len(df)}, unparseable: {df['unparseable'].mean():.1%}")

    ego_paths = {os.path.basename(p).split(".")[0]: p
                 for p in glob.glob(os.path.join(HERE, "test_egomotion", "*.egomotion.parquet"))}
    gt_cache = {}
    preds = {}
    for arm in mfig.ARMS:
        with open(os.path.join(HERE, f"rollouts_{arm}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r and r.get("rollouts") and all(r["rollouts"]):
                    preds[(arm, r["submission_key"])] = r["waypoints"]

    def gt_for(key, t0):
        if key not in gt_cache:
            gt_cache[key] = mfig.ego_frame_waypoints(
                pd.read_parquet(ego_paths[key.rsplit("_", 1)[0]]), t0)
        return gt_cache[key]

    ades = []
    for _, row in df.iterrows():
        gt = gt_for(row["key"], row["t0"])
        pred = np.array(preds[(row["arm"], row["key"])][row["ridx"]])
        ades.append(float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1).mean()))
    df["ade"] = ades

    d = df.dropna(subset=["faith"]).copy()
    ev = d.groupby("key").agg(faith=("faith", "mean"), ade=("ade", "mean"),
                              n=("faith", "size")).reset_index()
    x, y = ev["faith"].to_numpy(), ev["ade"].to_numpy()
    r = np.corrcoef(x, y)[0, 1]
    z, se = np.arctanh(r), 1 / np.sqrt(len(ev) - 3)
    print(f"scored rollouts: {len(d)}; events: {len(ev)}; "
          f"per-event r = {r:.3f}, 95% CI [{np.tanh(z-1.96*se):.2f}, {np.tanh(z+1.96*se):.2f}]")

    # within-event contrast, scenario difficulty removed
    d["ade_dm"] = d["ade"] - d.groupby("key")["ade"].transform("mean")
    hi = d[d["faith"] == 1.0]["ade_dm"]
    lo = d[d["faith"] < 1.0]["ade_dm"]
    print(f"within-event demeaned ADE: fully-consistent n={len(hi)} {hi.mean():+.3f} m "
          f"vs inconsistent n={len(lo)} {lo.mean():+.3f} m (diff {hi.mean()-lo.mean():+.3f})")
    per_rollout_r = np.corrcoef(d["faith"], d["ade_dm"])[0, 1]
    print(f"per-rollout within-event r = {per_rollout_r:.3f} (n={len(d)})")

    # ---- figure: full-scale panel (a) + existing panel (b) case ----
    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28,
                          left=0.06, right=0.97, top=0.93, bottom=0.19)
    ax = fig.add_subplot(gs[0])
    jit = np.random.default_rng(0).normal(0, 0.006, len(x))
    ax.scatter(x + jit, y, s=22, color="#3477b4", alpha=0.7, zorder=3)
    xs = np.linspace(x.min() - 0.03, x.max() + 0.03, 100)
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
    case_ev = ev[ev["key"] == mfig.CASE["key"]]
    if len(case_ev):
        ax.scatter(case_ev["faith"], case_ev["ade"], marker="^", s=130,
                   facecolors="none", edgecolors="red", linewidths=1.8, zorder=4)
    ax.set_xlabel("Faithfulness (offline consistency parser, event mean)", fontsize=11)
    ax.set_ylabel("Decision Error (ADE vs GT, m)", fontsize=11)
    ax.grid(alpha=0.3, ls="--")
    fig.text(0.29, 0.03,
             f"(a) Reasoning–Decision Correlation  (r = {r:.2f}, n = {len(ev)} events, "
             f"{len(d)} rollouts)", ha="center", fontsize=11.5)

    # panel (b): reuse the committed case study by re-running the maker's logic
    mfig_case_png = os.path.join(HERE, "misalignment_figure.png")
    axb = fig.add_subplot(gs[1])
    axb.set_axis_off()
    import textwrap
    arm, key, ridx = mfig.CASE["arm"], mfig.CASE["key"], mfig.CASE["ridx"]
    case_wp = np.array(preds[(arm, key)][ridx])
    case_txt = None
    with open(os.path.join(HERE, f"rollouts_{arm}.jsonl")) as f:
        for line in f:
            rr = json.loads(line)
            if rr["submission_key"] == key:
                case_txt = rr["rollouts"][ridx]
                t0c = rr["t0_us_used"]
                break
    gt = gt_for(key, t0c)
    v = np.linalg.norm(np.diff(case_wp[:, :2], axis=0), axis=1) / DT
    axb.text(0.5, 1.0, "Right Reasoning", ha="center", va="top", fontsize=13,
             weight="bold", transform=axb.transAxes)
    box_green = dict(boxstyle="round,pad=0.55", fc="#dcefd8", ec="none")
    axb.text(0.5, 0.87, textwrap.fill(f"Reasoning: “{case_txt}”", 58),
             ha="center", va="center", fontsize=10.5, style="italic",
             transform=axb.transAxes, bbox=box_green)
    jrec = next(j for j in map(json.loads, open(os.path.join(HERE, "judgments.jsonl")))
                if (j["arm"], j["key"], j["ridx"]) == (arm, key, ridx))
    axb.text(0.5, 0.70,
             textwrap.fill(f"Judge: expected “{jrec['expected_long']}” — "
                           f"{jrec['why']}", 62),
             ha="center", va="center", fontsize=9.5,
             transform=axb.transAxes, bbox=box_green)
    axb.annotate("", xy=(0.5, 0.50), xytext=(0.5, 0.60), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    axb.text(0.5, 0.47, "Wrong Decision", ha="center", va="top", fontsize=13,
             weight="bold", transform=axb.transAxes)
    axb.text(0.5, 0.35,
             textwrap.fill(f"Executed trajectory: accelerates {v[:5].mean():.1f} → "
                           f"{v[-5:].mean():.1f} m/s, advancing {v.sum()*DT:.1f} m into "
                           f"the crosswalk (GT remains stopped)", 62),
             ha="center", va="center", fontsize=10, color="#8b1a1a",
             transform=axb.transAxes,
             bbox=dict(boxstyle="round,pad=0.55", fc="#f6d7d4", ec="none"))
    axi = axb.inset_axes([0.22, 0.0, 0.56, 0.24])
    t = np.arange(1, len(case_wp)) * DT
    axi.plot(t, v, color="#c0392b", lw=2, label="model")
    axi.plot(t, np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1) / DT,
             color="#2e7d32", lw=2, ls="--", label="GT (stays stopped)")
    axi.set_xlabel("time (s)", fontsize=8)
    axi.set_ylabel("speed (m/s)", fontsize=8)
    axi.tick_params(labelsize=7)
    axi.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.text(0.75, 0.03, "(b) Misalignment", ha="center", fontsize=11.5)

    out = os.path.join(HERE, "misalignment_figure_full.png")
    fig.savefig(out, dpi=180)
    print("wrote", out)


if __name__ == "__main__":
    main()
