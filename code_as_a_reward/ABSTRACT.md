# Code as a Reward: Deterministic Verification of Chain-of-Causation Reasoning for Driving-Policy Post-Training

## Abstract

Vision-language-action driving policies such as Alpamayo 1.5 emit a
chain-of-causation (CoC) reasoning trace alongside each planned trajectory,
but nothing constrains that reasoning to be *faithful*: the policy can
claim a maneuver it does not execute, justify its action with an object
that is not in the scene, or stitch true observations to true actions with
a false "because." Existing reward signals for reasoning quality either
compare generated text to a reference annotation — rewarding parroting
rather than faithfulness — or ask a second LLM to judge the trace, which
is slow, nondeterministic, API-bound, and itself unverified. We propose
**code as a reward**: compiling each free-form CoC trace into small, typed,
machine-checkable claims and scoring them with deterministic programs over
the recorded scene state and the rollout's own predicted trajectory.

A rule-based parser (lexicons tuned on ~2,000 real Alpamayo CoC strings)
segments each trace into plan beats and extracts three claim types:
*commitment* claims ("I will slow down"), verified against kinematic
features of the trajectory the same rollout produced; *perceptual* claims
("a pedestrian is crossing"), verified against the dataset's
`obstacle.offline` 3D actor tracks (present for 97.4% of clips); and
*causal* claims linking the two, checked as their conjunction. On a
120-trace claim taxonomy, ~62% of claims are actor-checkable while ~27%
(cones, signals, road geometry) have no ground truth in the dataset —
so abstention is a first-class outcome: the reward is precision over
*decided* claims, with abstained claims excluded from the denominator,
coverage reported alongside, and a small penalty for text the parser
could not account for. Missing ground truth therefore never reads as
model unfaithfulness, and unverifiable prose cannot inflate the score.

We integrate the verifier as a drop-in reasoning reward for GRPO
post-training (cosmos-rl), keeping the validated trajectory-accuracy and
comfort components, gates, and mixing formula of the recipe unchanged so
that code-reward and LLM-judge runs differ in the reasoning signal only.
The reward is local CPU work (milliseconds per rollout, no API calls, no
network), removing the latency-driven failure modes observed with
judge-based rewards at ~14k calls per run. Every verdict is auditable:
each scored trace retains its per-claim pass/fail/abstain evidence.

The causal check is deliberately a conjunction test — it catches claimed
maneuvers that did not happen and justifications that were not there, but
not a false causal *link* between individually-true parts; testing the
link requires counterfactual rollouts we leave to future work. Verifier
thresholds and reward weights are documented first-cut defaults pending
calibration against hand-labeled traces; the first GRPO canary with this
reward is in flight.

## Pipeline at a glance

```
CoC text ──parse──▶ typed claims ──verify──▶ per-claim verdicts ──aggregate──▶ reward
                    commitment  ── vs TrajectoryFeatures (rollout's own trajectory)
                    perceptual  ── vs obstacle.offline actor tracks (scene)
                    causal      ── conjunction of the two
                                   PASS / FAIL / ABSTAIN, precision over decided
```

Code map: `coc_claim_parser.py` (parser), `obstacle_tracks.py` (scene
ground truth), `commitment_verifier.py` / `perceptual_verifier.py`
(per-claim checks), `trace_reward.py` (aggregation),
`../rl_posttrain/rewards/code_reward_entry.py` (GRPO integration).
