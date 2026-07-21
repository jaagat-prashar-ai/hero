# SPDX-License-Identifier: Apache-2.0
"""
obstacle_tracks.py — loads the upstream dataset's `obstacle.offline` actor
tracks (the ground-truth side of perceptual-claim verification) into a
queryable per-clip scene state.

Where the data comes from: nvidia/PhysicalAI-Autonomous-Vehicles ships an
`obstacle.offline` label per clip (present for 97.4% of clips — Phase 0
finding) as a table of timestamped 3D bounding boxes, fetched via the
`physical_ai_av` package's PhysicalAIAVDatasetInterface — the same vendored
entrypoint third_party/alpamayo1.5's load_physical_aiavdataset.py already
uses for egomotion, NOT a hand-rolled parquet walk over the repo layout.

Frame convention — the single most load-bearing fact in this module,
verified empirically against clip f0d61901-... (2026-07-21): every row has
reference_frame == 'rig' with reference_frame_timestamp_us equal to the
row's own timestamp_us, i.e. each detection is expressed in the EGO RIG
FRAME AT ITS OWN INSTANT. center_x is forward, center_y is lateral with
POSITIVE = LEFT (ISO 8855, the same convention classify_maneuvers.py and
commitment_verifier.py use). So "is there a pedestrian ahead of ego at
time t" is a direct sign test on that instant's rows — no egomotion
transform needed. The trade-off: positions are ego-RELATIVE snapshots, so
an actor's apparent motion mixes its own motion with ego's; anything that
needs an actor's world-frame kinematics (e.g. "is it really stationary")
must difference against egomotion, which this module deliberately leaves
to callers until a claim type actually needs it.

Python-version split — why the network fetch is a lazy import:
physical-ai-av requires Python >= 3.11 while this repo's project env pins
3.10 (see pyproject.toml). Everything below EXCEPT load_obstacle_tracks'
body is pure pandas/numpy and runs (and is tested) under the project env
against a committed real-data fixture; the physical_ai_av import happens
inside load_obstacle_tracks so merely importing this module never requires
the 3.11+ environment. Harvest-side code (which already runs alpamayo's
3.12 env on the cluster) is where the network path executes.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

# The exact upstream label-class vocabulary, from the same empirical dump
# that established the frame convention above. Kept as a module constant so
# the perceptual verifier's entity->class mapping can assert against it:
# a typo'd class name should fail loudly, not silently match zero tracks.
OBSTACLE_LABEL_CLASSES = frozenset(
    {"automobile", "person", "rider", "other_vehicle", "trailer", "protruding_object"}
)


@dataclasses.dataclass
class ActorTrack:
    """One tracked actor's samples over the clip, time-ordered.

    Arrays are parallel (one row per detection of this track). Positions
    are rig-frame-at-each-instant — see module docstring — so
    `centers_m[i]` answers "where was this actor RELATIVE TO EGO at
    `timestamps_us[i]`", which is exactly the shape perceptual claims come
    in ("a pedestrian ahead", "cones on the right side")."""

    track_id: str
    label_class: str  # one of OBSTACLE_LABEL_CLASSES
    timestamps_us: np.ndarray  # (N,) int64, strictly increasing
    centers_m: np.ndarray  # (N, 3) float64, ego-relative (x fwd, y left, z up)
    sizes_m: np.ndarray  # (N, 3) float64 box extents

    @property
    def first_us(self) -> int:
        return int(self.timestamps_us[0])

    @property
    def last_us(self) -> int:
        return int(self.timestamps_us[-1])

    def window(self, start_us: int, end_us: int) -> "ActorTrack":
        """This track restricted to samples with start_us <= t <= end_us
        (possibly zero-length — callers check len(), same 'empty is data,
        not an error' contract as pandas slicing). Perceptual claims are
        judged over a rollout's time window [t0, t0 + horizon], never the
        whole clip: a pedestrian who left the scene ten seconds before t0
        does not make 'a pedestrian is crossing' true."""
        mask = (self.timestamps_us >= start_us) & (self.timestamps_us <= end_us)
        return ActorTrack(
            track_id=self.track_id,
            label_class=self.label_class,
            timestamps_us=self.timestamps_us[mask],
            centers_m=self.centers_m[mask],
            sizes_m=self.sizes_m[mask],
        )

    def min_ego_distance_m(self) -> float:
        """Closest planar (xy) approach to ego across this track's samples.
        Planar on purpose: box centers carry a z offset (~half the actor's
        height above the road) that is irrelevant to 'how near was it'."""
        if len(self.timestamps_us) == 0:
            return float("inf")
        return float(np.linalg.norm(self.centers_m[:, :2], axis=1).min())

    def mean_bearing(self) -> tuple[float, float]:
        """Mean (x_forward_m, y_left_m) over this track's samples — the
        coarse 'where was it relative to ego' answer perceptual states
        like 'ahead' / on the 'right side' need. Mean rather than a single
        instant because claims describe the actor's general placement over
        the maneuver, and single-frame autolabel boxes jitter."""
        if len(self.timestamps_us) == 0:
            return (float("nan"), float("nan"))
        return (float(self.centers_m[:, 0].mean()), float(self.centers_m[:, 1].mean()))

    def apparent_speed_mps(self) -> float:
        """Mean speed of the actor's EGO-RELATIVE position over this
        track's samples. This is apparent motion: it conflates the actor's
        own motion with ego's (module docstring trade-off), so it can
        support relative-motion states ('approaching', 'pulling away',
        'crossing') but NOT world-frame ones ('stopped' — a parked car has
        large apparent speed while ego drives past it). Callers needing
        world-frame kinematics must difference against egomotion; nothing
        in this module does, and pretending otherwise here would bake a
        subtle wrongness into every downstream verdict."""
        if len(self.timestamps_us) < 2:
            return float("nan")
        dt_s = np.diff(self.timestamps_us).astype(np.float64) / 1e6
        steps_m = np.linalg.norm(np.diff(self.centers_m[:, :2], axis=0), axis=1)
        return float((steps_m / dt_s).mean())


@dataclasses.dataclass
class SceneObstacles:
    """All of one clip's obstacle.offline tracks, ready for perceptual
    queries. Construct via from_dataframe (pure, no network — the path
    unit tests use) or load_obstacle_tracks (streams from HF)."""

    clip_id: str
    tracks: list[ActorTrack]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, clip_id: str) -> "SceneObstacles":
        """Build from a raw obstacle.offline DataFrame (columns as shipped
        upstream: timestamp_us, track_id, label_class, center_*, size_*,
        reference_frame, ...).

        Validates the rig-frame assumption instead of trusting it: the
        module docstring's geometry claims are only true while
        reference_frame == 'rig' holds for every row, and a future dataset
        revision quietly switching to a world frame would otherwise turn
        every downstream ahead/left/distance answer into garbage with no
        error anywhere. Unknown label classes are rejected for the same
        reason (see OBSTACLE_LABEL_CLASSES)."""
        frames = set(df["reference_frame"].unique())
        if frames != {"rig"}:
            raise ValueError(
                f"clip {clip_id}: expected all obstacle rows in 'rig' frame, got {frames} — "
                "this module's ego-relative geometry assumption does not hold for this data"
            )
        classes = set(df["label_class"].unique())
        if not classes <= OBSTACLE_LABEL_CLASSES:
            raise ValueError(
                f"clip {clip_id}: unknown obstacle label classes {classes - OBSTACLE_LABEL_CLASSES} — "
                "update OBSTACLE_LABEL_CLASSES (and the perceptual verifier's entity mapping) deliberately"
            )

        tracks: list[ActorTrack] = []
        for track_id, group in df.groupby("track_id", sort=False):
            group = group.sort_values("timestamp_us")
            label_classes = group["label_class"].unique()
            # A track flip-flopping classes would make entity matching
            # ambiguous; upstream autolabels keep it constant per track, so
            # enforce that rather than silently taking the first.
            if len(label_classes) != 1:
                raise ValueError(
                    f"clip {clip_id} track {track_id}: inconsistent label classes {label_classes}"
                )
            tracks.append(
                ActorTrack(
                    track_id=str(track_id),
                    label_class=str(label_classes[0]),
                    timestamps_us=group["timestamp_us"].to_numpy(dtype=np.int64),
                    centers_m=group[["center_x", "center_y", "center_z"]].to_numpy(
                        dtype=np.float64
                    ),
                    sizes_m=group[["size_x", "size_y", "size_z"]].to_numpy(dtype=np.float64),
                )
            )
        # Deterministic order (by first appearance time, then id) so
        # downstream reports are stable across pandas groupby versions.
        tracks.sort(key=lambda t: (t.first_us, t.track_id))
        return cls(clip_id=clip_id, tracks=tracks)

    def actors_present(
        self,
        start_us: int,
        end_us: int,
        *,
        classes: set[str] | None = None,
        max_distance_m: float | None = None,
        min_samples: int = 2,
    ) -> list[ActorTrack]:
        """Tracks (windowed to [start_us, end_us]) that were actually
        present during the window: at least `min_samples` detections
        (default 2 — a single-frame detection is exactly the shape
        autolabel false positives take, and one sample also can't support
        any relative-motion state check), of the given classes (validated
        against OBSTACLE_LABEL_CLASSES — a typo should fail, not match
        nothing), within `max_distance_m` of ego at closest approach.

        This is THE question the perceptual verifier asks per claim:
        "was an actor of class X near ego during this rollout's window?"."""
        if classes is not None and not classes <= OBSTACLE_LABEL_CLASSES:
            raise ValueError(
                f"unknown obstacle classes {classes - OBSTACLE_LABEL_CLASSES}; "
                f"valid: {sorted(OBSTACLE_LABEL_CLASSES)}"
            )
        result: list[ActorTrack] = []
        for track in self.tracks:
            if classes is not None and track.label_class not in classes:
                continue
            windowed = track.window(start_us, end_us)
            if len(windowed.timestamps_us) < min_samples:
                continue
            if max_distance_m is not None and windowed.min_ego_distance_m() > max_distance_m:
                continue
            result.append(windowed)
        return result
