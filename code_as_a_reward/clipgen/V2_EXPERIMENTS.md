# ClipGen GT-only reward pipeline

The production pipeline has a strict offline/training boundary.

## 1. Offline reward construction

For each clip, the builder consumes only recorded scene observations, the
NVIDIA CoC, and the NVIDIA action at the authoritative training keyframe.
It never imports Alpamayo, samples a policy rollout, or reads a rollout cache.

The LLM emits a `clipgen.reward.v1` JSON specification. A deterministic
compiler supplies the executable implementation. The DSL enforces nonnegative
weights, a total budget of one, monotonic tolerant trajectory curves, bounded
perception-only credit, and at least one reasoning/action conjunction.

Offline publication requires:

- a valid GT target contract;
- intact GT score at least `0.70`;
- every applicable reasoning or action corruption at least `0.40` below the
  intact score;
- finite, truthful component accounting; and
- `policy_rollouts_used: false` in artifact provenance.

Launch Full1050 shards with:

```bash
bash code_as_a_reward/clipgen/configs/launch_clipgen_offline_gt.sh full
```

## 2. Corpus packaging

`build_reward_corpus.py` validates the offline artifact provenance and gates,
recompiles every JSON spec, and materializes a versioned corpus containing only
published functions plus `corpus_manifest.json`. Arbitrary legacy Python is
never copied into the corpus.

## 3. GRPO-time verification

The policy samples 12 reasoning/trajectory pairs. The clip's frozen cached
function scores all 12 and is itself the GRPO reward—there is no second exact-GT
ADE mixture. The worker selects argmax, or top-3 when the ablation flag is set,
and re-runs the same function on independently corrupted reasoning and action
variants.

A group contributes reward only when the selected positive scores at least
`0.70`, every corruption drops by at least `0.40`, and the score distribution
has useful rank resolution. A failed group is neutralized to zero advantage.

## 4. Repair without perturbation overfitting

Failed live groups are written as immutable replay records. Repair is
asynchronous: it requires failures from multiple groups, shows only the
development split to the LLM, validates against GT again, and opens a sealed
group once. Accepted repairs are proposals for a later checkpoint/epoch
boundary; a reward never changes inside a GRPO batch.

The old rollout-sampling curation entrypoint remains only for explicit
diagnostics and ablations. It is not used by the production offline builder.
