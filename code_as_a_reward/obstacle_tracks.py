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
