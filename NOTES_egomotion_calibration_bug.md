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

## Fix direction (drafted and validated below; not yet applied to source)

- `_to_bytes()` in `build_webdataset.py` needs an explicit branch for
  `Interpolator` objects — extract real timestamped egomotion state into a
  DataFrame before `.to_parquet()`, instead of relying on the
  `json.dumps(default=str)` fallback.
- `_RigidTransform.__repr__` (`build_wds_worker.py:73`) or `_serialize()`
  (`build_webdataset.py:327`) needs to call `self.rotation.as_quat()` /
  `.as_matrix()` explicitly rather than embedding `self.rotation!r}`.

## Drafted fix (validated against real `physical_ai_av`, not yet applied)

Downloaded the real `physical_ai_av==0.2.0` wheel directly from PyPI
(`files.pythonhosted.org`, since `pip index`/`pip download` couldn't resolve
it through the sandboxed network here) and inspected the source directly:

- `physical_ai_av/egomotion.py`: `EgomotionState` is a dataclass with
  `pose: RigidTransform`, `velocity`, `acceleration`, `curvature` fields, and
  a `from_egomotion_df(df)` classmethod that expects columns
  `qx,qy,qz,qw,x,y,z,vx,vy,vz,ax,ay,az,curvature` — i.e. this *is* the
  original raw schema the data came from.
- `physical_ai_av/utils/interpolation.py`: `Interpolator` exposes
  `.timestamps` (raw int64 µs array) and `.values` (an `EgomotionState`
  whose fields are full arrays over those timestamps, not a single sample).
- `physical_ai_av/calibration.py`: `SensorExtrinsics.sensor_poses` is a
  plain `dict[str, RigidTransform]`; `RigidTransform.rotation` /
  `.translation` are directly accessible attributes.

This means the fix doesn't need to reconstruct or resample anything — the
raw arrays are already sitting on the object, just inaccessible through
`repr()`/`str()`.

**Proposed `_egomotion_to_bytes()`** (replaces the `_to_bytes(egomotion)`
call site at `build_webdataset.py:311`):

```python
def _egomotion_to_bytes(egomotion: Any) -> bytes:
    values = egomotion.values
    xyz = np.asarray(values.pose.translation)
    quat = np.asarray(values.pose.rotation.as_quat())
    vel = np.asarray(values.velocity)
    acc = np.asarray(values.acceleration)
    curv = np.asarray(values.curvature).reshape(-1)
    df = pd.DataFrame({
        "timestamp_us": np.asarray(egomotion.timestamps),
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "qx": quat[:, 0], "qy": quat[:, 1], "qz": quat[:, 2], "qw": quat[:, 3],
        "vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2],
        "ax": acc[:, 0], "ay": acc[:, 1], "az": acc[:, 2],
        "curvature": curv,
    })
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()
```

**Proposed `_serialize_calibration_value()`** (replaces the inline
`_serialize()` closure at `build_webdataset.py:327-332`):

```python
def _serialize_calibration_value(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _serialize_calibration_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_calibration_value(v) for v in obj]
    if isinstance(obj, spt.Rotation):
        return obj.as_quat().tolist()
    if isinstance(obj, spt.RigidTransform):
        return {
            "rotation_quat_xyzw": np.asarray(obj.rotation.as_quat()).tolist(),
            "translation": np.asarray(obj.translation).tolist(),
        }
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _serialize_calibration_value(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if hasattr(obj, "to_dict"):
        return _serialize_calibration_value(obj.to_dict())
    return obj
```

Requires two new imports in `build_webdataset.py`: `import dataclasses` and
`import scipy.spatial.transform as spt`.

**Validation performed** (no HF/cluster access needed — pure logic test):
installed the real wheel into a throwaway venv (`python3.11 -m venv`),
built real `EgomotionState` / `SensorExtrinsics` / `Interpolator` objects
from it, and ran both functions above against them:
- Egomotion parquet round-trips: reconstructed rotation matrices match the
  originals exactly (compared via rotation matrices, not raw quaternion
  components, since quaternions have a sign ambiguity — `q` and `-q`
  represent the same rotation).
- `EgomotionState.from_egomotion_df(roundtrip_df)` recovers a state
  matching the original, confirming schema compatibility with the
  package's own expected input format.
- Calibration rotation data round-trips correctly (quaternion + translation,
  not a memory address) under **both** scipy states: the production stub
  `RigidTransform` (`build_wds_worker.py`'s workaround, active when scipy
  lacks a real `RigidTransform` — confirmed true for the pinned
  `scipy>=1.15.0` resolving to 1.15.3 locally) and a real scipy
  `RigidTransform` (present in scipy 1.17.1, installed by default in the
  throwaway venv) — so the fix doesn't depend on which one is active.

Not yet applied to `build_webdataset.py` — deliberately holding off so the
actual code change lands as its own separate, reviewable commit.

// We also want to exploer howe can build off ffmpeg with libsvtav1 compiled in, since the cluster's current ffmpeg (UBunut 22.04's stock page) only has libaom-av1. 

There are three concrete ways (infer which one is the best from the diagonistic run). We can enable UBunutu's jammy-backports pocket, then apt-get install ffmpeg, sometims backportws carries a newer ffmpeg build with more codecs. This is a one-line change in ensure_ffmpeg_av1(): add the backports source, get apt-get update, reinstall ffmpeg, re-check for lbsvtav1. What is the jammy backports pocket? 
Proposed solution: the job's pip installs (from PyPI) and HF downloads worked fine in thes esame logs,s o general HTTPs egress to the public internet is allowed. It's specifically apt that'st restricted to the internal mirror. This means we could download a prebuilt static ffmpeg binary with libstav1 baked in over HTTPS (bypassing apt) instead of relying on the system package manager. 6.3x faster, quality within noise (PSNR 45.83 dB vs 45.49 dB). NOw implementing this as an automatic runtime fallback. Now, the order is: check system ffmpeg own encoder list (if libstav1 is already there, use it immediately and no download). If not, try downlaoding the static ffmpeg build with libsvtav1 (this ios what fires on the cluster every tuime). If that download fails for any reason, the fall ack is to whartver the system has (libaom-av1). Last resport is genertic apt-get installf fmpeg attempt, then raise. 

// Clips to chunking?

The dataset: 306,152 clips packed into 3,146 chunks (~97 clips/chunk average). Each chunk is a shared file per feature (egomotion, 3 calibration files, 7 camera files ≈ 11 files/chunk) — HF's client caches these to local disk, so once a chunk is downloaded, every other clip in that chunk is free (local disk read, zero more HF requests).

The problem: build_webdataset.py's partitioning (work[split][rank::world_size], line 678) assigns clips round-robin by sorted clip ID — with no relationship to chunk. Once world_size gets anywhere near the average chunk size (~97), every chunk's clips get scattered across a different node for almost every clip, and since each node has its own separate local disk cache, the same chunk file gets re-downloaded independently by dozens of nodes instead of once.

The math:
- Ideal (grouped by chunk): ~11 files/chunk × 3,146 chunks ≈ 34,600 total HF file downloads for the entire dataset, ever — trivial against the 5000-req/5-min budget.
- Current scheme at world_size≈100: effectively back to ~11 files/clip × 306,152 clips ≈ 3.37 million downloads — ~97x more traffic than necessary. That's roughly 56+ hours of HF traffic alone at the rate limit, which lines up with the 429 failures already seen in the earlier 100-way run.

So before tuning worker counts, the actual fix is: partition by chunk, not by clip. Want me to implement that (group clip IDs by chunk from avdi.clip_index, assign whole chunks round-robin to ranks instead of individual clips)? That's the change that actually makes worker-count tuning meaningful — right now almost any large world_size blows the HF budget regardless of what number we pick.

