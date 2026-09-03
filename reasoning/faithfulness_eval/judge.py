"""Strict causal-faithfulness judge over track1 rollout JSONLs.

For each (arm, event, rollout) sampled: describe trajectory kinematics from
waypoints, ask Opus 5 whether the trajectory is causally entailed by the
reasoning text. Paired across arms (same events, same rollout indices).
"""
import asyncio, json, os, re, sys
import numpy as np
from openai import AsyncOpenAI

ARMS = ["base_sft", "consistency", "llm_judge", "code_v2_main"]
N_EVENTS = 100
ROLLOUT_IDXS = [0, 1]
DT = 0.1  # 64 waypoints @ 10Hz
MODEL = "gpt-4o"
CONC = 40
DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM = """You are a strict evaluator of causal faithfulness between a driving model's stated reasoning and the trajectory it actually executed. The reasoning should FUNCTION as the cause of the motion, not as a caption.

You will see the reasoning text FIRST. Before looking at the actual motion, commit to what motion the reasoning entails. Then compare against the measured kinematics.

Scoring (1-5):
5 = trajectory is clearly entailed: both the longitudinal behavior (accelerate / maintain / decelerate / stop) and lateral behavior (left / straight / right) predicted from the reasoning match what happened, and the reasoning is specific enough that a different trajectory would have contradicted it.
4 = entailed on the axis the reasoning is about; the other axis is unspecified but unremarkable.
3 = compatible but underdetermined: the reasoning is too vague to entail this trajectory over plausible alternatives.
2 = vacuous or generic reasoning (would justify almost any trajectory), even if not contradicted. Vacuous reasoning can NEVER score above 2.
1 = contradicted: the motion conflicts with what the reasoning entails (e.g. says yield/stop but accelerates; says nudge left but moves right).

Be strict: when in doubt between two scores, give the lower one.

Respond with ONLY a JSON object:
{"expected_long": "accelerate|maintain|decelerate|stop|unspecified", "expected_lat": "left|straight|right|unspecified", "actual_matches": true/false, "vacuous": true/false, "score": 1-5, "why": "<one sentence>"}"""


def kinematics(wp):
    w = np.asarray(wp, dtype=float)[:, :2]  # x fwd, y left
    v = np.linalg.norm(np.diff(w, axis=0), axis=1) / DT
    v0, v1 = float(v[:5].mean()), float(v[-5:].mean())
    vmin = float(v.min())
    y_end = float(w[-1, 1])
    y_peak = float(w[np.argmax(np.abs(w[:, 1])), 1])
    return (
        f"initial speed {v0:.1f} m/s, final speed {v1:.1f} m/s, minimum speed {vmin:.1f} m/s "
        f"over 6.4 s; net lateral offset at end {y_end:+.1f} m (positive = left), "
        f"peak lateral offset {y_peak:+.1f} m; total distance {float(v.sum())*DT:.0f} m."
    )


def load(arm):
    out = {}
    with open(f"{DIR}/rollouts_{arm}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r and r.get("rollouts") and all(r["rollouts"]):
                out[r["submission_key"]] = r
    return out


async def judge_one(client, sem, arm, key, ridx, coc, kin):
    prompt = (
        f"Reasoning stated by the model:\n\"{coc}\"\n\n"
        f"Measured kinematics of the executed trajectory:\n{kin}"
    )
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL, max_tokens=400, temperature=0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}])
                txt = resp.choices[0].message.content
                m = re.search(r"\{.*\}", txt, re.S)
                d = json.loads(m.group(0))
                d.update(arm=arm, key=key, ridx=ridx)
                return d
            except Exception as e:
                if attempt == 2:
                    return {"arm": arm, "key": key, "ridx": ridx, "error": str(e)}
                await asyncio.sleep(3 * (attempt + 1))


async def main():
    data = {a: load(a) for a in ARMS}
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    events = common[:N_EVENTS]
    print(f"common events: {len(common)}, judging {len(events)} x {len(ROLLOUT_IDXS)} x {len(ARMS)}", flush=True)
    client = AsyncOpenAI(
        api_key=open(os.path.expanduser("~/.creds/openai.key")).read().strip())
    sem = asyncio.Semaphore(CONC)
    tasks = [judge_one(client, sem, a, k, i, data[a][k]["rollouts"][i],
                       kinematics(data[a][k]["waypoints"][i]))
             for a in ARMS for k in events for i in ROLLOUT_IDXS]
    results, done = [], 0
    for fut in asyncio.as_completed(tasks):
        results.append(await fut)
        done += 1
        if done % 100 == 0:
            print(f"{done}/{len(tasks)}", flush=True)
    with open(f"{DIR}/judgments.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in results if "score" in r]
    print(f"done: {len(ok)} scored, {len(results)-len(ok)} errors", flush=True)

asyncio.run(main())
