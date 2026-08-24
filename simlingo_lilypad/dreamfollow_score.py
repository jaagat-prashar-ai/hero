"""Programmatic scoring for the dreamer instruction-following eval.

For each paired sample the model saw a counterfactual dreamer instruction; the
predictions JSONL carries its predicted waypoints plus (via eval_infos) BOTH
the instructed trajectory (new_wps) and the scene's default trajectory
(org_wps). Compliance is measured without any LLM:

  ADE_instr  = mean L2(pred, instructed waypoints)
  ADE_org    = mean L2(pred, default waypoints)
  margin     = ADE_org - ADE_instr   (positive => tracked the instruction)
  follow     = ADE_instr < ADE_org   (on rows where the two GTs differ)

Rows where instructed == default (the sampler drew the 'org' option) can't
discriminate and are excluded from follow-rate/margin, but still count toward
plain ADE_instr. Breakdowns per dreamer mode (stop, faster, lane change, ...).

Usage: python simlingo_lilypad/dreamfollow_score.py --pred-dir <dir> --out <summary.json>
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def ade(a: np.ndarray, b: np.ndarray) -> float:
    t = min(len(a), len(b))
    return float(np.linalg.norm(a[:t] - b[:t], axis=1).mean())


def bootstrap_ci(vals, iters=10000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(iters))
    return [means[int(0.025 * iters)], means[int(0.975 * iters)]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    arms = sorted(p.parent.name for p in pred_dir.glob("*/predictions.jsonl"))
    assert arms, f"no <arm>/predictions.jsonl under {pred_dir}"

    per_arm: dict[str, list[dict]] = {}
    for arm in arms:
        rows = []
        for line in (pred_dir / arm / "predictions.jsonl").open():
            r = json.loads(line)
            info = r["eval_infos"]
            pred = np.array(r["waypoints_pred"], dtype=float)
            instr = np.array(info["new_wps"], dtype=float)
            org = np.array(info["org_wps"], dtype=float)
            t = min(len(instr), len(org))
            discriminative = not np.allclose(instr[:t], org[:t], atol=1e-3)
            rows.append({
                "path": r["path"],
                "mode": info.get("mode"),
                "ade_instr": ade(pred, instr),
                "ade_org": ade(pred, org),
                "discriminative": discriminative,
            })
        per_arm[arm] = rows

    # pairing sanity: identical clips AND identical instruction draws per index
    ref = next((a for a in arms if "original" in a), arms[0])
    for arm in arms:
        assert [r["path"] for r in per_arm[arm]] == [r["path"] for r in per_arm[ref]], \
            f"clip order differs: {arm} vs {ref}"
        assert [r["mode"] for r in per_arm[arm]] == [r["mode"] for r in per_arm[ref]], \
            f"instruction draws differ: {arm} vs {ref} — pairing broken"

    summary: dict = {}
    disc_idx = [i for i, r in enumerate(per_arm[ref]) if r["discriminative"]]
    summary["_meta"] = {"n_rows": len(per_arm[ref]), "n_discriminative": len(disc_idx)}

    for arm in arms:
        rows = per_arm[arm]
        d = [rows[i] for i in disc_idx]
        margins = [r["ade_org"] - r["ade_instr"] for r in d]
        summary[arm] = {
            "ade_instr_all": round(float(np.mean([r["ade_instr"] for r in rows])), 4),
            "ade_instr_disc": round(float(np.mean([r["ade_instr"] for r in d])), 4),
            "ade_org_disc": round(float(np.mean([r["ade_org"] for r in d])), 4),
            "follow_rate": round(float(np.mean([r["ade_instr"] < r["ade_org"] for r in d])), 4),
            "follow_rate_ci95": [round(x, 4) for x in bootstrap_ci([float(r["ade_instr"] < r["ade_org"]) for r in d])],
            "margin_mean_m": round(float(np.mean(margins)), 4),
            "margin_ci95": [round(x, 4) for x in bootstrap_ci(margins)],
        }
        by_mode = defaultdict(list)
        for r in d:
            by_mode[r["mode"]].append(float(r["ade_instr"] < r["ade_org"]))
        summary[arm]["follow_rate_by_mode"] = {
            m: {"n": len(v), "rate": round(float(np.mean(v)), 3)} for m, v in sorted(by_mode.items())
        }

    for arm in arms:
        if arm == ref:
            continue
        deltas = [
            (per_arm[ref][i]["ade_instr"] - per_arm[arm][i]["ade_instr"])
            for i in disc_idx
        ]
        summary[f"paired_ade_improvement:{arm}-vs-{ref}"] = {
            "n_pairs": len(deltas),
            "mean_m": round(float(np.mean(deltas)), 4),
            "ci95": [round(x, 4) for x in bootstrap_ci(deltas, seed=1)],
            "note": "positive = arm tracks the instruction more closely than the baseline",
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
