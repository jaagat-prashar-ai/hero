# SPDX-License-Identifier: Apache-2.0
"""Generate a Track-1 PhysicalAI AV reasoning submission from an Alpamayo 1.5 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EXPECTED_KEYS = 284
ROLLOUTS_PER_KEY = 6


def _write_atomic(path: Path, payload: dict[str, list[str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def validate_submission(payload: dict[str, list[str]]) -> None:
    if len(payload) != EXPECTED_KEYS:
        raise ValueError(f"expected {EXPECTED_KEYS} keys, got {len(payload)}")
    bad = {
        key: value
        for key, value in payload.items()
        if not isinstance(value, list)
        or len(value) != ROLLOUTS_PER_KEY
        or any(not isinstance(text, str) or not text.strip() for text in value)
    }
    if bad:
        raise ValueError(f"{len(bad)} keys do not contain six non-empty reasoning strings")


def main() -> None:
    import numpy as np
    import torch
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from reasoning.wds_test_loader import iter_test_samples

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--max-generation-attempts", type=int, default=5)
    args = parser.parse_args()

    output = Path(args.output)
    results: dict[str, list[str]] = {}
    if output.exists():
        results = json.loads(output.read_text())

    model = Alpamayo1_5.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)

    for sample_idx, sample in enumerate(iter_test_samples(args.shards)):
        key = sample["submission_key"]
        if key in results and len(results[key]) == ROLLOUTS_PER_KEY:
            continue
        messages = helper.create_message(
            frames=sample["image_frames"].flatten(0, 1),
            camera_indices=sample["camera_indices"],
        )
        tokenized = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = helper.to_device(
            {
                "tokenized_data": tokenized,
                "ego_history_xyz": sample["ego_history_xyz"],
                "ego_history_rot": sample["ego_history_rot"],
            },
            "cuda",
        )
        texts: list[str] = []
        for attempt in range(args.max_generation_attempts):
            needed = ROLLOUTS_PER_KEY - len(texts)
            if needed <= 0:
                break
            torch.cuda.manual_seed_all(args.seed + sample_idx * 100 + attempt)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                _xyz, _rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    num_traj_samples=needed,
                    max_generation_length=256,
                    return_extra=True,
                )
            generated = [
                str(x).strip()
                for x in np.asarray(extra["cot"], dtype=object)[0].reshape(-1)
            ]
            texts.extend(text for text in generated if text)
            if len(texts) < ROLLOUTS_PER_KEY:
                print(
                    f"retrying {key}: {len(texts)}/{ROLLOUTS_PER_KEY} valid CoCs "
                    f"after attempt {attempt + 1}",
                    flush=True,
                )
        texts = texts[:ROLLOUTS_PER_KEY]
        if len(texts) != ROLLOUTS_PER_KEY:
            raise RuntimeError(
                f"{key}: only {len(texts)}/{ROLLOUTS_PER_KEY} non-empty CoCs after "
                f"{args.max_generation_attempts} attempts"
            )
        results[key] = texts
        _write_atomic(output, results)
        print(f"submission progress: {len(results)}/{EXPECTED_KEYS} ({key})", flush=True)

    validate_submission(results)
    _write_atomic(output, results)
    print(f"SUBMISSION_COMPLETE {output} keys={len(results)}", flush=True)


if __name__ == "__main__":
    main()
