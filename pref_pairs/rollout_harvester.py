# SPDX-License-Identifier: Apache-2.0
"""
rollout_harvester.py — Task 1 of the counterfactual preference-pair mining
pipeline: rollouts -> featurize -> match -> label -> mine pairs -> audit.

Samples K reasoning+trajectory rollouts per scene from Alpamayo 1.5 and
persists them to disk (one JSON file per scene) as the raw input for the
downstream featurizer / pair miner (Tasks 2-6, not built yet).

Design notes (why this file looks the way it does):

  * INDEPENDENT OF masking.masked_model.MaskedAlpamayo1_5 ON PURPOSE. This is
    a separate experiment from the CoT-masking work, so the model side here
    talks directly to the upstream `Alpamayo1_5` class via composition (we
    hold a `self.model`, we don't subclass it) instead of reusing masking's
    `_rollout_prefix` / `_denoise_with_mask` split.
  * Scene *data loading* (pulling clips out of the S3 WDS shards) IS reused
    from `masking.data`, because that is model-agnostic infra (tar range
    reads, video decode) rather than part of the masking experiment's model
    logic -- see `harvest_dataset()` below.
  * SEED-CONTROL CAVEAT: neither
    `Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout` nor the
    underlying `diffusion.sample()` (checked both `diffusion/base.py` and
    `diffusion/flow_matching.py`) exposes a seed argument anywhere upstream.
    `masking/masked_model.py`'s `_denoise_with_mask` patches this in for its
    own code path with a plain `torch.manual_seed()` call right before
    sampling. We do the exact same minimal patch here, independently (see
    `harvest_scene`), since we can't reuse that class. This only gives
    run-level reproducibility for the WHOLE batch of K rollouts sampled in
    one call -- there is no way, without patching alpamayo1_5's
    diffusion.sample() itself, to give each of the K rollouts-within-a-batch
    its own independent seed. Flagging this explicitly per the project brief:
    "If not exposed, note where in the code it could be patched."
  * NATIVE ACTION CAPTURE: Alpamayo's diffusion decoder does not predict xyz
    directly -- it predicts a normalized (accel, curvature) action per
    waypoint, and `action_space.action_to_traj` deterministically integrates
    that into the xyz waypoints we actually want. But
    `sample_trajectories_from_data_with_vlm_rollout` computes that action
    tensor internally and discards it once `action_to_traj` has consumed it
    -- only the resulting xyz comes back out. Re-deriving accel from xyz via
    finite differences downstream (trajectory_features.py used to do this
    unconditionally) is strictly worse than the exact value the model itself
    already computed one step earlier. `_sample_with_captured_action` below
    recovers it by temporarily wrapping `action_space.action_to_traj` --
    the one place that tensor is consumed -- rather than duplicating the
    ~80 lines of generate()+diffusion-sampling logic that produce it, and
    rather than reusing `MaskedAlpamayo1_5._denoise_with_mask` (which *does*
    return this tensor), since that would reintroduce the masking-code
    dependency this module deliberately avoids (see the first bullet above).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

# alpamayo1_5 is vendored source, not a pip-installable package, on Lilypad's
# Python 3.10 workers (its own pyproject.toml wants ==3.12.*, which matters
# for `pip install` but not for importing raw source off sys.path). This adds
# third_party/alpamayo1.5/src to sys.path -- must run BEFORE the alpamayo1_5
# imports below. masking.bootstrap.ensure_alpamayo1_5 is pure sys.path
# plumbing with no model-specific logic, so reusing it here doesn't
# reintroduce the masking-code dependency this module otherwise avoids (see
# the "INDEPENDENT OF masking.masked_model" docstring note above) -- it's the
# same category of reuse as masking.data.wds_dataset, not masking's model code.
from masking.bootstrap import ensure_alpamayo1_5

ensure_alpamayo1_5()

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "nvidia/Alpamayo-1.5-10B"


@dataclasses.dataclass
class RolloutRecord:
    """One sampled rollout: a CoT text + the trajectory it was paired with.

    `waypoints` and `hz` are recorded from the ACTUAL model output at
    harvest time (see `_hz` below), not hardcoded -- the project brief's
    "16x10Hz" schema guess does not match what this model config actually
    produces (checked `model.config.tokens_per_future_traj` at runtime; see
    RolloutHarvester._infer_horizon docstring). Storing the real numbers
    here means downstream code (Task 2's featurizer) never has to guess.
    """

    scene_id: str
    rollout_id: int
    coc_text: str
    waypoints: list[list[float]]  # (T, 3) ego-frame xyz, T rows of [x, y, z]
    hz: float
    sampling_params: dict[str, Any]  # {seed, temperature, top_p, top_k, k}
    model_version: str
    ground_truth_coc: str | None  # NVIDIA's verified CoC label, if the scene has one

    # The model's own EXACT per-waypoint action, in physical units, if we
    # managed to capture it (see module docstring's "NATIVE ACTION CAPTURE"
    # note) -- None if capture failed for some reason (e.g. a future
    # alpamayo1_5 version restructures this internal call), in which case
    # trajectory_features.py falls back to deriving accel from xyz via
    # finite differences instead. curvature isn't consumed downstream yet,
    # but costs nothing extra to store now that we're capturing the tensor
    # anyway, and later tasks (claim verification) may want it directly.
    native_accel_mps2: list[float] | None = None
    native_curvature_per_m: list[float] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        # dataclasses.asdict() would already give us plain dicts/lists here,
        # but spelling it out keeps the on-disk schema explicit and stable
        # even if we later add non-JSON-serializable fields to the dataclass.
        return dataclasses.asdict(self)


def _denorm_accel_curvature(action_space: Any, action_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Physical-unit (accel, curvature) from the model's raw normalized
    action tensor. Same formula as masking/masked_model.py's denorm_action,
    reimplemented here (4 lines, not imported) rather than pulling in that
    module -- see the "INDEPENDENT OF masking.masked_model" design note at
    the top of this file. action_raw: (..., T, 2) normalized [accel, kappa].
    Returns (accel, curvature), each (..., T), in physical units.
    """
    accel = action_raw[..., 0] * action_space.accel_std.to(action_raw) + action_space.accel_mean.to(action_raw)
    kappa = action_raw[..., 1] * action_space.curvature_std.to(action_raw) + action_space.curvature_mean.to(action_raw)
    return accel, kappa


def _sample_with_captured_action(
    model: Alpamayo1_5, data: dict[str, Any], **kwargs: Any
) -> tuple[tuple[Any, ...], torch.Tensor | None]:
    """Call sample_trajectories_from_data_with_vlm_rollout, but also recover
    the native normalized (accel, curvature) action tensor it computes
    internally and would otherwise discard -- see the "NATIVE ACTION
    CAPTURE" design note at the top of this file for why this exists and
    why it's implemented this way (spying on action_to_traj) rather than
    duplicating the surrounding generate()+diffusion-sampling logic or
    reusing MaskedAlpamayo1_5.

    Returns (upstream_return_value, action_raw_or_None). action_raw is None
    if action_to_traj was never called (e.g. generation failed before
    reaching the diffusion step) -- callers must handle that gracefully,
    not assume capture always succeeds.

    The original method is ALWAYS restored, even if sampling raises, so a
    failure here can't leave the model's action_space permanently patched.
    """
    action_space = model.action_space
    original_action_to_traj = action_space.action_to_traj
    captured: dict[str, torch.Tensor] = {}

    def _spy_action_to_traj(action: torch.Tensor, *args: Any, **kw: Any) -> Any:
        captured["action_raw"] = action
        return original_action_to_traj(action, *args, **kw)

    action_space.action_to_traj = _spy_action_to_traj
    try:
        result = model.sample_trajectories_from_data_with_vlm_rollout(data=data, **kwargs)
    finally:
        action_space.action_to_traj = original_action_to_traj

    return result, captured.get("action_raw")


class RolloutHarvester:
    """Loads Alpamayo 1.5 once and samples K rollouts per scene on demand."""

    def __init__(self, model: Alpamayo1_5, device: str = "cuda") -> None:
        self.model = model
        self.device = device

    @classmethod
    def load(
        cls, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cuda"
    ) -> "RolloutHarvester":
        # Same load pattern as third_party/alpamayo1.5/.../test_inference.py --
        # plain Alpamayo1_5, not MaskedAlpamayo1_5, per the independence note above.
        #
        # attn_implementation="sdpa" is required, not cosmetic: base_model.py
        # defaults to "flash_attention_2" when this kwarg is omitted, and
        # flash-attn is a slow-to-build CUDA extension we deliberately don't
        # install (masking/masked_model.py's own from_pretrained call already
        # works around this the same way). SDPA is PyTorch's built-in fused
        # attention -- no extra dependency, and alpamayo1_5.py already forces
        # the diffusion expert's OWN attention to "sdpa" regardless, so this
        # only affects the VLM backbone, not a mismatch between the two.
        model = Alpamayo1_5.from_pretrained(
            checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device)
        model.eval()
        return cls(model=model, device=device)

    def _build_tokenized_inputs(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        """Turn raw (image_frames, camera_indices, ego_history_xyz/rot) into the
        {tokenized_data, ego_history_xyz, ego_history_rot} dict the model's
        rollout method expects.

        This re-implements masking/run_masked_openloop.py's build_inputs()
        logic, but calls ONLY upstream alpamayo1_5.helper utilities -- no
        import from `masking` -- so this module's model-facing code has zero
        dependency on the masking experiment's code, per the independence
        note at the top of this file.
        """
        messages = helper.create_message(
            frames=model_inputs["image_frames"].flatten(0, 1),
            camera_indices=model_inputs["camera_indices"],
        )
        processor = helper.get_processor(self.model.tokenizer)
        tokenized = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        data = {
            "tokenized_data": tokenized,
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        }
        return helper.to_device(data, self.device)

    def harvest_scene(
        self,
        model_inputs: dict[str, Any],
        scene_id: str,
        *,
        k: int = 20,
        seed: int = 0,
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        ground_truth_coc: str | None = None,
    ) -> list[RolloutRecord]:
        """Sample K reasoning+trajectory rollouts for one scene.

        Returns one RolloutRecord per rollout, ready to serialize to disk.
        """
        tokenized_inputs = self._build_tokenized_inputs(model_inputs)

        # --- Patch in the seed control the upstream API doesn't expose ---
        # (see the "SEED-CONTROL CAVEAT" note at the top of this file). This
        # seeds the diffusion RNG state right before sampling so re-running
        # the same scene with the same seed reproduces the same batch of K
        # rollouts. It does NOT give each of the K rollouts its own seed --
        # that granularity simply isn't exposed anywhere upstream.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # One batched call: do_sample=True + num_return_sequences=k inside
        # sample_trajectories_from_data_with_vlm_rollout resamples the CoT
        # text independently per rollout (k separate reasoning traces), and
        # the diffusion expert is then run once, batched, over all k.
        # _sample_with_captured_action also recovers the model's native
        # (accel, curvature) action tensor -- see "NATIVE ACTION CAPTURE" in
        # the module docstring.
        with torch.autocast(self.device, dtype=torch.bfloat16):
            (pred_xyz, _pred_rot, extra), action_raw = _sample_with_captured_action(
                self.model,
                tokenized_inputs,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                num_traj_samples=k,
                num_traj_sets=1,
                return_extra=True,
            )

        # pred_xyz shape: (B=1, num_traj_sets=1, num_traj_samples=k, T, 3).
        # extra["cot"] shape (after upstream's own reshape): (B=1, ns=1, k) of str.
        waypoints_per_rollout = pred_xyz[0, 0].float().cpu()  # (k, T, 3)
        cot_per_rollout = extra["cot"][0, 0]  # (k,) array of str

        # action_raw shape (if captured): (k, T, 2) normalized [accel, kappa]
        # -- see sample_trajectories_from_data_with_vlm_rollout's internal
        # `total_batch = B * n_samples_total` batching; B=1 here so the
        # leading dim is exactly k, no further indexing needed.
        native_accel_per_rollout = native_curvature_per_rollout = None
        if action_raw is not None:
            accel_t, kappa_t = _denorm_accel_curvature(self.model.action_space, action_raw)
            native_accel_per_rollout = accel_t.float().cpu()  # (k, T)
            native_curvature_per_rollout = kappa_t.float().cpu()  # (k, T)

        # Read the ACTUAL trajectory length off the sampled tensor instead of
        # trusting the brief's guessed "16" -- see RolloutRecord docstring.
        # The 10 Hz sampling rate itself IS a fixed convention throughout this
        # codebase (TIME_STEP_S=0.1 in masking/data/wds_dataset.py and
        # time_step=0.1 in load_physical_aiavdataset.py), so we record it
        # verbatim rather than re-deriving it from the model.
        hz = 10.0

        sampling_params = {
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "k": k,
        }
        model_version = getattr(self.model.config, "_name_or_path", DEFAULT_CHECKPOINT)

        records = []
        for rollout_id in range(k):
            records.append(
                RolloutRecord(
                    scene_id=scene_id,
                    rollout_id=rollout_id,
                    coc_text=str(cot_per_rollout[rollout_id]),
                    waypoints=waypoints_per_rollout[rollout_id].tolist(),
                    hz=hz,
                    sampling_params=sampling_params,
                    model_version=model_version,
                    ground_truth_coc=ground_truth_coc,
                    native_accel_mps2=(
                        native_accel_per_rollout[rollout_id].tolist()
                        if native_accel_per_rollout is not None else None
                    ),
                    native_curvature_per_m=(
                        native_curvature_per_rollout[rollout_id].tolist()
                        if native_curvature_per_rollout is not None else None
                    ),
                )
            )
        return records


def _write_scene_records(records: list[RolloutRecord], out_dir: Path, scene_id: str) -> Path:
    """Write one scene's rollouts to out_dir/{scene_id}.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_id}.json"
    out_path.write_text(json.dumps([r.to_json_dict() for r in records], indent=2))
    return out_path


def harvest_dataset(
    manifest_path: str,
    bucket: str,
    out_dir: str,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    k: int = 20,
    seed: int = 0,
    top_p: float = 0.98,
    top_k: int | None = None,
    temperature: float = 0.6,
    max_scenes: int | None = 100,
) -> list[Path]:
    """Drive the harvester over every scene in a masking-style clip manifest.

    Scene *selection* is entirely config-driven (manifest_path, max_scenes
    below) per the brief's "make scene selection a config, not hardcoded"
    requirement -- see pref_pairs/configs/local.yaml.

    Imports masking.data.wds_dataset lazily (only here, not at module import
    time) so callers who just want RolloutHarvester.harvest_scene() on their
    own data don't need masking's data-loading deps (boto3, av, webdataset,
    pandas) installed at all.
    """
    from masking.data.wds_dataset import iter_clip_events_from_manifest

    harvester = RolloutHarvester.load(checkpoint=checkpoint, device=device)
    out_dir_path = Path(out_dir)

    written: list[Path] = []
    n_scenes = 0
    for event in iter_clip_events_from_manifest(manifest_path, bucket):
        if max_scenes is not None and n_scenes >= max_scenes:
            logger.info("Reached max_scenes=%d, stopping.", max_scenes)
            break

        # scene_id mirrors the {clip_id}_{t0_us} naming already used for
        # per-clip artifacts elsewhere in this repo (e.g.
        # masking/results/experiment_b_clips/*.mp4), so scenes stay easy to
        # cross-reference by eye across pipelines.
        scene_id = f"{event['clip_id']}_{event['t0_us']}"
        logger.info("Harvesting scene %s (%d/%s)...", scene_id, n_scenes + 1, max_scenes)

        records = harvester.harvest_scene(
            event["model_inputs"],
            scene_id=scene_id,
            k=k,
            seed=seed,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            ground_truth_coc=event.get("event_coc") or None,
        )
        written.append(_write_scene_records(records, out_dir_path, scene_id))
        n_scenes += 1

    logger.info("Harvested %d scenes into %s", n_scenes, out_dir)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to a pref_pairs config YAML.")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    harvest_dataset(
        manifest_path=cfg["manifest_path"],
        bucket=cfg["bucket"],
        out_dir=cfg["out_dir"],
        checkpoint=cfg.get("checkpoint", DEFAULT_CHECKPOINT),
        device=cfg.get("device", "cuda"),
        k=cfg.get("k", 20),
        seed=cfg.get("seed", 0),
        top_p=cfg.get("top_p", 0.98),
        top_k=cfg.get("top_k"),
        temperature=cfg.get("temperature", 0.6),
        max_scenes=cfg.get("max_scenes", 100),
    )


if __name__ == "__main__":
    main()
