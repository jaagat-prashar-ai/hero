# Investigation notes — egomotion/calibration rotation serialization bug (unfixed)

Status: **confirmed, not yet fixed**. Blocks relaunching the WDS build.
Related: BUGS.md (S3 upload fix), this is a separate data-corruption issue
found by inspecting shard *contents* after that fix landed.

## What was checked

Downloaded and directly inspected the same clip
(`00003c6c-37db-4276-a698-5d8dd5095a3e`) from both:
- `s3://research-datasets-chicago/nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds-smoke/train/shard_000_00000.tar` (pre-av1, npz cameras, ~1.3GB)
- `s3://research-datasets-chicago/nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds-smoke-av1/train/shard_000_00000.tar` (av1 mp4 cameras, ~8.9MB)

Both bugs below are present, byte-identical, in both shards — i.e. **not**
caused by the av1 mp4 camera-encoding change, which is independently
verified good (all 7 camera streams present in both, no drop).

## Bug 1 — `egomotion.parquet` is not a parquet file

Entire 64-byte content in both shards (md5 `05acab6de4651f81c99ebe862cc97a2b`):

    "Interpolator[EgomotionState](time_range=[(-196163, 70053634)])"

No timestamps, positions, or velocities — a JSON-encoded `repr()` of a
Python object saved with a `.parquet` extension. `pd.read_parquet()` on it
throws `ArrowInvalid: Parquet magic bytes not found in footer`.

**Root cause:** `build_wds/data/build_webdataset.py:223` `_to_bytes()` tries
`obj.to_parquet()` first. The `EGOMOTION` feature returned by
`avdi.get_clip_feature()` is an `Interpolator` object with no `to_parquet`
method, so it falls through to a last-resort `json.dumps(obj, default=str)`
branch that just stringifies the whole object.

## Bug 2 — `calibration.json`'s `sensor_extrinsics` rotations are lost

Translations and the rest of calibration (`camera_intrinsics`,
`vehicle_dimensions`) are intact and correct. But every camera pose in
`sensor_extrinsics.sensor_poses` looks like:

    "camera_front_wide_120fov": "RigidTransform(rotation=<scipy.spatial.transform._rotation.Rotation object at 0x7fa6a269f000>, translation=array([ 2.16349745, -0.04690826,  1.6044457 ]))"

The translation is real; the rotation is a bare memory address, unique per
process, carrying zero actual rotation data.

**Root cause:** `build_wds/data/build_wds_worker.py:52-75` stubs a
`RigidTransform` class (needed because NVIDIA's `physical_ai_av` expects one
that doesn't exist in released scipy — confirmed locally: scipy 1.15.3 has
no `RigidTransform`). Its `__repr__` at line 73 does
`f"RigidTransform(rotation={self.rotation!r}, ...)"`. `self.rotation` is a
raw `scipy.spatial.transform.Rotation`, which has no custom `__repr__`, so
Python's default `object.__repr__` kicks in and prints only the memory
address. This repr gets captured because `build_webdataset.py`'s
calibration `_serialize()` closure (line 327-332) falls back to `str(obj)`
for anything without `.to_dict()`/`__dict__`, invoked per-leaf by the
top-level `json.dumps(cal, default=str)` at line 339.

Note: a real (non-stub) scipy `RigidTransform` would **not** fix this on
its own if it relied on the same repr fallback — scipy's `Rotation` object
still has no data-revealing `__repr__` on the versions this repo pins
(`scipy>=1.15.0`, which resolved to 1.15.3 with no `RigidTransform` at all
in the production run). The real fix needs an explicit `.as_quat()` /
`.as_matrix()` call before serialization, not reliance on `str()`/`repr()`.

## Not investigated / separately noted

`feature_presence` in the per-clip `.json` metadata claims
`lidar_top_360fov: True`, but neither shard actually contains a
`lidar_top_360fov.parquet` file. Unrelated to the two bugs above — worth
checking separately if LiDAR data is needed. Don't trust `feature_presence`
flags as proof a file's content is usable; verify by extracting and
inspecting directly.

## Fix direction (not yet implemented)

- `_to_bytes()` in `build_webdataset.py` needs an explicit branch for
  `Interpolator` objects — extract real timestamped egomotion state into a
  DataFrame before `.to_parquet()`, instead of relying on the
  `json.dumps(default=str)` fallback.
- `_RigidTransform.__repr__` (`build_wds_worker.py:73`) or `_serialize()`
  (`build_webdataset.py:327`) needs to call `self.rotation.as_quat()` /
  `.as_matrix()` explicitly rather than embedding `self.rotation!r}`.
