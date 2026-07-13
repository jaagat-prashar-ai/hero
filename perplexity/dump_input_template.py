# T1.3: capture the exact input-sequence template the demo feeds the model.
#
# We do NOT reimplement prompt construction. This script calls the exact same
# vendored code path test_inference.py uses (helper.create_message,
# processor.apply_chat_template, model.fuse_traj_tokens) and simply captures
# the intermediate tensors, instead of letting them disappear inside
# AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout. `fuse_traj_tokens`
# is a public method (see alpamayo/src/alpamayo_r1/models/base_model.py:168),
# so calling it directly (rather than only through the wrapper) is calling
# real vendored code, not reimplementing it.
#
# Two things get produced per sample:
#   raw_input_ids   -- straight out of apply_chat_template. History positions
#                      are still the single placeholder token <|traj_history|>,
#                      repeated 48x (see helper.create_message).
#   fused_input_ids -- after model.fuse_traj_tokens(): the 48 placeholder ids
#                      have been replaced with real quantized history-bin
#                      tokens computed from ego_history_xyz/rot. This exact
#                      tensor is what alpamayo_r1.py's
#                      sample_trajectories_from_data_with_vlm_rollout hands to
#                      self.vlm.generate(input_ids=...) -- i.e. this is the
#                      literal prefix "fed to the model".
#
# Note what does NOT change between raw and fused: the vision-token span.
# Qwen3-VL's image placeholder tokens (image_token_id) are expanded by the
# processor itself when it builds pixel_values/image_grid_thw during
# apply_chat_template, before fuse_traj_tokens ever runs. fuse_traj_tokens only
# ever touches occurrences of the single <|traj_history|> id.

import json
import sys
from dataclasses import asdict, dataclass

import torch
from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.token_utils import to_special_token

from s3_clip_loader import load_clip_from_s3_extract

CHECKPOINT = "nvidia/Alpamayo-R1-10B"
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"  # same clip as alpamayo's test_inference.py
S3_CLIP_DIR = (
    "/tmp/claude-4035/-home-jaagat-prashar-workspace-research-project-template-main-perplexity/"
    "64d82f1e-0858-4573-8406-df96512c11e1/scratchpad/s3/clip"
)
T0_US = 5_100_000
FIXTURE_PATH = "fixtures/t1_3_input_template.json"

# Marker token names we care about for teacher forcing later (T2). All of
# these are read off the live tokenizer below -- none are hardcoded ids.
MARKER_NAMES = [
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_history_start",
    "traj_history",
    "traj_history_end",
    "traj_future_start",
    "traj_future",
    "traj_future_end",
]


@dataclass
class Span:
    """Half-open token index range [start, end) within an input_ids sequence."""

    start: int
    end: int
    length: int


def load_hf(clip_id: str, t0_us: int) -> dict:
    return load_physical_aiavdataset(clip_id, t0_us=t0_us)


def load_s3(clip_id: str, t0_us: int) -> dict:
    return load_clip_from_s3_extract(S3_CLIP_DIR, clip_id, t0_us=t0_us)


DATA_LOADERS = {"hf": load_hf, "s3": load_s3}


def load_model_and_sample(data_loader) -> tuple[AlpamayoR1, dict]:
    """Load the checkpoint and one raw dataset sample, exactly as test_inference.py does."""
    data = data_loader(CLIP_ID, T0_US)
    model = AlpamayoR1.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).to("cuda")
    return model, data


def build_prompt(model: AlpamayoR1, data: dict) -> dict:
    """Build the literal prompt tensors, capturing both pre- and post-fusion input_ids."""
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    raw_input_ids = inputs["input_ids"].clone()

    # This is the exact call alpamayo_r1.py's sample_trajectories_from_data_with_vlm_rollout
    # makes (line 164) before handing input_ids to self.vlm.generate(). We call it directly
    # so we can see the result instead of only its downstream effect on generation.
    fused_input_ids = model.fuse_traj_tokens(
        inputs["input_ids"].clone(),
        {"ego_history_xyz": data["ego_history_xyz"].to("cuda"), "ego_history_rot": data["ego_history_rot"].to("cuda")},
    )

    aux = {k: v for k, v in inputs.items() if k != "input_ids"}
    return {
        "raw_input_ids": raw_input_ids,
        "fused_input_ids": fused_input_ids,
        "aux": aux,
    }


def token_id_span(ids: torch.Tensor, start_id: int, end_id: int) -> Span:
    """Find the half-open span [first start_id, first end_id after it]."""
    ids = ids[0]
    start_positions = (ids == start_id).nonzero(as_tuple=True)[0]
    end_positions = (ids == end_id).nonzero(as_tuple=True)[0]
    if len(start_positions) == 0 or len(end_positions) == 0:
        raise ValueError(f"start_id={start_id} or end_id={end_id} not found in sequence")
    start = start_positions[0].item()
    end = end_positions[0].item() + 1  # inclusive of the end marker itself
    return Span(start=start, end=end, length=end - start)


def vision_span(ids: torch.Tensor, vlm_config) -> Span:
    """Find the span bounded by vision_start_token_id / vision_end_token_id."""
    return token_id_span(ids, vlm_config.vision_start_token_id, vlm_config.vision_end_token_id)


def ego_history_span(ids: torch.Tensor, traj_token_ids: dict) -> Span:
    return token_id_span(ids, traj_token_ids["history_start"], traj_token_ids["history_end"])


def marker_token_ids(tokenizer) -> dict:
    """Read the actual vocab id for every marker token name off the live tokenizer."""
    return {name: tokenizer.convert_tokens_to_ids(to_special_token(name)) for name in MARKER_NAMES}


def build_fixture(model: AlpamayoR1, data: dict, prompt: dict, data_source: str) -> dict:
    raw_ids = prompt["raw_input_ids"]
    fused_ids = prompt["fused_input_ids"]
    tokenizer = model.tokenizer

    v_span = vision_span(raw_ids, model.vlm.config)
    h_span = ego_history_span(raw_ids, model.config.traj_token_ids)
    markers = marker_token_ids(tokenizer)

    # Everything that is neither vision nor the history block is "text prompt"
    # (system + user instruction text + the trailing <|cot_start|> that opens
    # the assistant turn for continuation).
    seq_len = raw_ids.shape[1]
    covered = set(range(v_span.start, v_span.end)) | set(range(h_span.start, h_span.end))
    text_positions = [i for i in range(seq_len) if i not in covered]

    fixture = {
        "clip_id": CLIP_ID,
        "t0_us": T0_US,
        "checkpoint": CHECKPOINT,
        "data_source": data_source,
        "seq_len": seq_len,
        "raw_input_ids": raw_ids[0].tolist(),
        "fused_input_ids": fused_ids[0].tolist(),
        "raw_tokens_decoded": tokenizer.convert_ids_to_tokens(raw_ids[0].tolist()),
        "vision_span": asdict(v_span),
        "ego_history_span": asdict(h_span),
        "text_prompt_positions": text_positions,
        "aux_shapes": {k: list(v.shape) for k, v in prompt["aux"].items() if isinstance(v, torch.Tensor)},
        "marker_token_ids": markers,
        "note": (
            "marker_token_ids are vocab ids for the reasoning/action boundary tokens. "
            "cot_end / meta_action_start / meta_action_end / traj_future_start do NOT "
            "appear in this prompt -- they only appear once the model has generated "
            "reasoning. This fixture covers the PREFIX only, per T1.3's scope."
        ),
    }
    if data_source == "s3":
        fixture["caveats"] = (
            "Sourced from our own S3 WDS mirror (shard_019_00000.tar), not HF directly, "
            "because HF's Xet CDN was hanging on this session. Egomotion is exact (raw "
            "floats, not lossy). Camera video was AV1-transcoded (crf=32) by "
            "build_wds/data/build_webdataset.py before upload, and per-frame capture "
            "timestamps were never stored in the WDS shard, so frames here are picked by "
            "index (last N), not by the official timestamp-based selection. This does NOT "
            "affect input_ids fidelity (vision-token span length depends on image "
            "resolution/count, not pixel content or capture time) but pixel_values "
            "themselves are not byte-identical to what the true HF path would produce. "
            "See s3_clip_loader.py's module docstring."
        )
    return fixture


def reconstruct_and_verify(model: AlpamayoR1, fixture: dict, data_loader) -> None:
    """Done-when check: rebuild fused_input_ids from raw data alone and diff byte-for-byte."""
    data = data_loader(fixture["clip_id"], fixture["t0_us"])
    prompt = build_prompt(model, data)
    rebuilt = prompt["fused_input_ids"][0].tolist()
    expected = fixture["fused_input_ids"]
    if rebuilt != expected:
        first_diff = next(i for i, (a, b) in enumerate(zip(rebuilt, expected)) if a != b)
        raise AssertionError(
            f"Reconstruction mismatch at position {first_diff}: "
            f"rebuilt={rebuilt[first_diff]} expected={expected[first_diff]}"
        )
    print(f"OK: reconstructed {len(rebuilt)} fused input_ids byte-for-byte from raw data alone.")


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "hf"
    if source not in DATA_LOADERS:
        raise SystemExit(f"unknown source {source!r}, expected one of {list(DATA_LOADERS)}")
    data_loader = DATA_LOADERS[source]
    fixture_path = FIXTURE_PATH if source == "hf" else FIXTURE_PATH.replace(".json", f"_{source}.json")

    model, data = load_model_and_sample(data_loader)
    prompt = build_prompt(model, data)
    fixture = build_fixture(model, data, prompt, data_source=source)

    print(f"source={source}")
    print(f"seq_len={fixture['seq_len']}")
    print(f"vision_span={fixture['vision_span']}")
    print(f"ego_history_span={fixture['ego_history_span']}")
    print(f"text_prompt token count={len(fixture['text_prompt_positions'])}")
    print(f"marker_token_ids={fixture['marker_token_ids']}")
    print(f"aux tensor shapes={fixture['aux_shapes']}")

    with open(fixture_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Saved fixture to {fixture_path}")

    reconstruct_and_verify(model, fixture, data_loader)


if __name__ == "__main__":
    main()
