# SPDX-License-Identifier: Apache-2.0
"""Real Alpamayo rollout sampling for clipgen's generation-time gate.

Fixes the GT-overfit bug found via d50ad0's select-then-verify diagnostic
(see BUGS.md / code_as_a_reward/clipgen/gate.py's docstring): the
generation-time gate previously only tested corruptions of GT (the one
recorded expert trajectory), so an LLM-written reward function could pass
by hardcoding GT's exact numbers (e.g. "heading change within 1 deg of
18.1") instead of a generalizable check -- such a function then scores
EVERY real GRPO rollout identically, since none reproduce GT's exact
numbers. The fix: gate against a real Alpamayo rollout group's argmax
instead (run_prototype.py wires this in), so the gate only accepts
functions that discriminate among trajectories the policy can actually
produce.

Same checkpoint training itself starts from (rl_posttrain/configs/
code_reward_full_cluster.yaml: alpamayo_model: "nvidia/Alpamayo-1.5-10B"),
same load/inference call as code_as_a_reward/ood_eval/worker.py's
load_model/run_model_rollout (that module's own docstring: "mirrors
third_party/alpamayo1.5/src/alpamayo1_5/test_inference.py's example
exactly"). ood_eval/worker.py samples num_traj_samples=1 per clip (a single
model-vs-GT comparison); this module bumps that to a whole group (default
12, matching the GRPO rollout-group size the training path dumps -- see
rl_posttrain/rewards/code_reward_entry.py's _maybe_dump_rollouts) so the
gate has an argmax to select and verify.

torch/alpamayo1_5 imports are lazy (inside the functions), same convention
as ood_eval/worker.py: this module is importable in the base env (no
torch), and only actually calling load_model/sample_rollout_group requires
running inside the bootstrapped Python 3.12 Alpamayo venv (see
code_as_a_reward/ood_eval/bootstrap_venv.py).
"""

from __future__ import annotations

from typing import Any

MODEL_CHECKPOINT = "nvidia/Alpamayo-1.5-10B"
HZ = 10.0  # load_physical_aiavdataset's time_step=0.1s convention
DEFAULT_GROUP_SIZE = 12


def fetch_clip_data(clip_id: str, t0_us: int, avdi=None):
    """One HF fetch (image frames + egomotion history/future, already in the
    ego-frame-at-t0 convention). Identical to ood_eval/worker.py's helper of
    the same name."""
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    return load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)


def load_model():
    """Load Alpamayo-1.5-10B once for the whole generation run.
    attn_implementation="eager" skips flash-attn (compiles from source,
    20-40+ min) -- same tradeoff as ood_eval/worker.py:load_model."""
    import torch
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    return Alpamayo1_5.from_pretrained(
        MODEL_CHECKPOINT, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")


def sample_rollout_group(
    model, data: dict, n: int = DEFAULT_GROUP_SIZE
) -> list[dict[str, Any]]:
    """Sample `n` (reasoning, trajectory) rollouts for one clip's scene in a
    SINGLE forward pass (num_traj_samples=n, not n separate calls).

    Returns a list shaped exactly like the training path's rollout-dump JSON
    (rl_posttrain/rewards/code_reward_entry.py's _maybe_dump_rollouts):
    [{"rollout_id": int, "coc_text": str, "waypoints": list[[x,y,z], ...]}],
    so it drops directly into
    code_as_a_reward.clipgen.analyze_group_rollouts.select_and_verify.
    """
    import torch
    from alpamayo1_5 import helper

    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=n,
            max_generation_length=256,
            return_extra=True,
        )
    # (B=1, n_traj_group=1, n, T, 3) -> (n, T, 3); ood_eval/worker.py's
    # single-sample call indexes [0, 0, 0] for the same reason.
    pred_xyz_np = pred_xyz.detach().float().cpu().numpy()[0, 0]
    cot_list = extra["cot"]
    return [
        {"rollout_id": i, "coc_text": cot_list[i], "waypoints": pred_xyz_np[i].tolist()}
        for i in range(n)
    ]


def sample_rollout_group_for_clip(
    model, clip_id: str, t0_us: int, n: int = DEFAULT_GROUP_SIZE, avdi=None
) -> list[dict[str, Any]]:
    """Convenience wrapper: fetch + sample in one call."""
    data = fetch_clip_data(clip_id, t0_us, avdi=avdi)
    return sample_rollout_group(model, data, n=n)
