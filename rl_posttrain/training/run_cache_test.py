import sys
import types
import zipfile


sys.modules.setdefault("ray", types.ModuleType("ray"))

from rl_posttrain.training.run import _CAMERA_SUBPARTS, _camera_chunk_covers_clips


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
