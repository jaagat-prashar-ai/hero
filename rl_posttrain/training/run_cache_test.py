import sys
import subprocess
import types
import zipfile

import pytest


sys.modules.setdefault("ray", types.ModuleType("ray"))

from rl_posttrain.training.run import (
    _CAMERA_SUBPARTS,
    _camera_chunk_covers_clips,
    _patch_toml,
    _run_streamed,
)


def _write_camera_zips(root, clip_ids, *, omit_timestamp_for=None):
    for camera in _CAMERA_SUBPARTS:
        path = root / "camera" / camera / f"{camera}.chunk_0007.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            for clip_id in clip_ids:
                archive.writestr(f"{clip_id}.{camera}.mp4", b"video")
                if (clip_id, camera) != omit_timestamp_for:
                    archive.writestr(
                        f"{clip_id}.{camera}.timestamps.parquet", b"timestamps"
                    )


def test_camera_chunk_coverage_requires_every_selected_clip(tmp_path) -> None:
    _write_camera_zips(tmp_path, {"clip-a", "clip-b"})
    assert _camera_chunk_covers_clips(tmp_path, 7, {"clip-a", "clip-b"})
    assert not _camera_chunk_covers_clips(tmp_path, 7, {"clip-a", "clip-c"})


def test_camera_chunk_coverage_requires_timestamps(tmp_path) -> None:
    missing = ("clip-a", _CAMERA_SUBPARTS[0])
    _write_camera_zips(tmp_path, {"clip-a"}, omit_timestamp_for=missing)
    assert not _camera_chunk_covers_clips(tmp_path, 7, {"clip-a"})


def test_run_streamed_times_out_hung_process() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        _run_streamed(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            timeout_s=0.05,
        )


def test_patch_toml_sets_exact_resume_checkpoint(tmp_path) -> None:
    template = tmp_path / "template.toml"
    template.write_text(
        """
[train]
output_dir = "old"
epoch = 1
resume = false
[train.train_policy]
kl_beta = 0.0
[policy]
model_name_or_path = "old-model"
[policy.parallelism]
dp_shard_size = 1
[logging]
project_name = "old-project"
experiment_name = "old-experiment"
[validation]
enable = false
freq = 10
n_generation = 1
"""
    )
    output = tmp_path / "patched.toml"
    resume = tmp_path / "resume" / "policy"
    _patch_toml(
        template,
        output,
        output_dir=tmp_path / "outputs",
        model_dir=tmp_path / "model",
        dp_shard_size=4,
        epoch=1,
        wandb_project="project",
        wandb_experiment="experiment",
        resume_checkpoint=resume,
    )

    import tomlkit

    patched = tomlkit.parse(output.read_text())
    assert patched["train"]["resume"] == str(resume)
