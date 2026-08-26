from rl_posttrain.training.audit_wds_coverage import REQUIRED_SUFFIXES, _member_suffix


def test_required_members_are_frame_accurate() -> None:
    assert "egomotion.parquet" in REQUIRED_SUFFIXES
    assert sum(name.endswith(".mp4") for name in REQUIRED_SUFFIXES) == 4
    assert sum(name.endswith(".timestamps.parquet") for name in REQUIRED_SUFFIXES) == 4


def test_member_suffix_rejects_next_clip() -> None:
    clip_id = "clip-a"
    assert _member_suffix("clip-a.camera_front_wide_120fov.mp4", clip_id) == (
        "camera_front_wide_120fov.mp4"
    )
    assert _member_suffix("clip-b.json", clip_id) is None
