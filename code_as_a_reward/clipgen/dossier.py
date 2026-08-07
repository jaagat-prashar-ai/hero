# SPDX-License-Identifier: Apache-2.0
"""Build the per-clip "dossier": the compact scene summary the generator reads.

Inputs are the same ground truth the code reward already uses -- a clip's
obstacle tracks (SceneObstacles) and the expert ego trajectory
(TrajectoryFeatures) -- reduced to ~20-30 lines of structured text. The
dossier deliberately contains the ACTUAL NUMBERS of the scene (times,
distances, speed drops) so the generator can derive scene-specific
thresholds instead of hand-tuned globals.

First-cut heuristics (documented for later calibration, same convention as
RewardConfig): tracks ranked by closest approach, capped at MAX_TRACKS;
bearing buckets at +/-20 deg (ahead) and +/-120 deg (behind).
"""
from __future__ import annotations
import dataclasses
import math
import numpy as np

# SceneObstacles: the input type describing all of the obstacles in the clip
from code_as_a_reward.obstacle_tracks import SceneObstacles # this is the obstacle.offline stuff
# TrajectoryFeatures: the input type describing the ego vehicle's trajectory 
# extract_features: a helper function that builds a TrajectoryFeatures object from raw waypoints 
from pref_pairs.trajectory_features import TrajectoryFeatures, extract_features
    # this is the trajectory featuarization thing we made in pref pairs from before? Double check this. 

# maximum number of obstacle tracks to include in the dossier text
MAX_TRACKS = 12

# Different datasets/exports name the ego-motion 
# Tuple for all column-name pairs we know how to recognize, in order of prefernce 
_EGOMOTION_XY_CANDIDATES = (("x", "y"), ("pox_x", "pos_y"), ("position_x", "position_y"), ("tx", "ty"))


@dataclasses.dataclass
class TrackSummary:
    """One obstacle track reduced to the facts a reward function needs."""

    # Unique identifier string for this obstacle track.
    track_id: str
    # The type/class of the obstacle (e.g. "vehicle", "pedestrian"). automobile, etc.
    label_class: str
    # Time (in seconds) when this obstacle first becomes visible in the clip.
    t_enter_s: float
    # Time (in seconds) when this obstacle stops being visible in the clip.
    t_exit_s: float
    # The smallest distance (in meters) between the ego vehicle and this obstacle, over the whole track.
        # how do we think about this with the world rails assumption from SimLingo (e.g., other ego agents)? Do we worry about this?
    closest_approach_m: float
    # The timestamp (in seconds) at which that closest approach happened. What are a good array of examples for closest approaches?
    t_closest_s: float
    # Rough direction bucket ("ahead"/"left"/"right"/"behind") of the obstacle relative to the ego, at closest approach.
    bearing_at_closest: str  # "ahead" | "left" | "right" | "behind"

# Given an obstacle's x (forward) and y (left) position relative to the ego,
# return a coarse compass-style bucket describing where it is.
def _bearing(x: float, y: float) -> str:
    """Bearing bucket in the ego frame (x fwd, y left)."""
    # atan2(y, x) gives the angle of the obstacle around the ego, in radians;
    # convert to degrees since the thresholds below are in degrees.
    # first we can begin by getting the angle between the agent and the obstacle
    angle = math.degrees(math.atan2(y, x))
    # if this angle is within 20 degrees relative to the agent, this should be ahead (i.e,. in our V span)
    # Within 20 degrees of straight ahead (either side) counts as "ahead".
    if abs(angle) <= 20.0:
        return "ahead"
    # beyond 120 degrees, we can say this is behind
    if abs(angle) >= 120.0:
        return "behind"
    # otherwise, it's left or right (positive angle means left and negative means right)
    return "left" if angle > 0 else "right"

# Take a full scene's worth of obstacle tracks and reduce it to a short ranked list of the most relevant ones (closest approach first).

def summarize_tracks(scene: SceneObstacles) -> list[TrackSummary]:
    """Rank tracks by the closest approach and keep the MAX_TRACKS nearest"""
    # This list will collect one TrackSummary per obstacle track. 
    out: list[TrackSummary] = []

    # Loop over every obstacle track that is present in the scene (e.g., from obstacle.offline from HF)
    for tr in scene.tracks: 
        # Compute the ego-relative distance (in the ground plane, x/y only) at every timestep of this track. 
        dists = np.linalg.norm(tr.centers_m[:, :2], axis=1)
        # Find the index of the timestep where the distance is smallest (closest approach). 
        i = int(np.argmin(dists))
        # Convert this track's timestamps from microseconds to seconds
        ts = tr.timestamps_us.astype(np.float64) / 1e6
        # Build a TrackSummary for this track using the values computed above 
        out.append(
            TrackSummary(
                track_id=str(tr.track_id),
                label_class=tr.label_class,
                # First timestamp in the track = where it enters view
                t_enter_s = float(ts[0]),
                # Last timestamp in the track = when it enters view
                t_exit_s = float(ts[-1]),
                # Distance at the closest-approach index. 
                closest_approach_m=float(dists[i]),
                # Time at the closest-approach index. 
                t_closest_s = float(ts[i]),
                # Direction bucket at the closets-approach index (x, y position at the moment)
                bearing_at_closest = _bearing(float(tr.centers_m[i, 0]), float(tr.ceners_m[i, 1])),
            )
        )



    # Sort all tracks so the one with the smallest closest-approach distance comes first.
    out.sort(key=lambda t: t.closest_approach_m)
    # Only keep the nearest MAX_TRACKS tracks; drop the rest.
    return out[:MAX_TRACKS]


# Convert a raw egomotion dataframe (arbitrary timestamps) into evenly-spaced
# (x, y) waypoints sampled at a fixed rate `hz`.
def waypoints_from_egomotion(df, hz: float = 10.0) -> np.ndarray:
    """(N, 2) planar waypoints resampled at `hz` from a raw egomotion frame.

    Tolerant of the column-name variants seen across the PAI egomotion
    exports. extract_features does NOT rotate into the t=0-heading frame --
    it trusts the input convention; the PAI egomotion logs are already
    ego-aligned at t=0 (verified across the smoke clips 2026-08-05: initial
    headings all within 1.4 deg of zero), so raw XY is fine here.
    """
    # Figure out which column holds timestamps: prefer "timestamp", else fall back to "timestamp_us".
    ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_us"
    # Try each known (x, y) column-name pair in order, and use the first one that's present.
    for cx, cy in _EGOMOTION_XY_CANDIDATES:
        if cx in df.columns and cy in df.columns:
            break
    # If none of the candidate pairs matched, we can't proceed - raise a clear error.
    else:
        raise ValueError(f"egomotion frame has no known XY columns (got {list(df.columns)})")
    # Pull the timestamp column out as a plain float array.
    ts = df[ts_col].to_numpy(dtype=np.float64)
    # Shift timestamps so they start at 0, and convert microseconds to seconds if the values look like microseconds.
    ts = (ts - ts[0]) / (1e6 if ts.max() > 1e6 else 1.0)  # us -> s when needed
    # Build a new evenly-spaced time grid from 0 up to the last timestamp, spaced 1/hz seconds apart.
    grid = np.arange(0.0, float(ts[-1]), 1.0 / hz)
    # Linearly interpolate the x position onto the new evenly-spaced time grid.
    x = np.interp(grid, ts, df[cx].to_numpy(dtype=np.float64))
    # Linearly interpolate the y position onto the new evenly-spaced time grid.
    y = np.interp(grid, ts, df[cy].to_numpy(dtype=np.float64))
    # Combine x and y into a single (N, 2) array of waypoints and return it.
    return np.stack([x, y], axis=1)


# Turn a TrajectoryFeatures object (the expert driver's trajectory) into a
# short list of plain-English summary lines.
def ego_lines(traj: TrajectoryFeatures) -> list[str]:
    """Human/LLM-readable summary of what the expert driver actually did."""
    # Convert speed-over-time to a plain float array for easy math below.
    speed = np.asarray(traj.speed_mps, dtype=np.float64)
    # Find the time (in seconds) at which speed was lowest, using the index of the minimum times the timestep size.
    t_min = float(np.argmin(speed)) * traj.dt_s
    # How much speed dropped, from the initial speed to the lowest speed reached.
    drop = traj.initial_speed_mps - traj.min_speed_mps
    # Convert lateral (sideways) offset-over-time to a plain float array.
    lat = np.asarray(traj.lateral_offset_m, dtype=np.float64)
    # Total distance traveled = sum of speed at every step * the time per step.
    dist = float(np.sum(speed) * traj.dt_s)
    # Build the list of summary lines, one per fact about the trajectory.
    lines = [
        # Line 1: how long the clip is, the sampling interval, and total distance covered.
        f"duration: {traj.n_waypoints * traj.dt_s:.1f} s at dt={traj.dt_s:.2f} s, distance {dist:.1f} m",
        # Line 2: how speed changed from start, to its minimum (and when), to the end.
        f"speed: {traj.initial_speed_mps:.1f} m/s initial -> min {traj.min_speed_mps:.1f} m/s"
        f" at t={t_min:.1f} s -> {traj.final_speed_mps:.1f} m/s final (drop {drop:.1f} m/s)",
        # Line 3: how far sideways the car ended up, and the largest sideways offset at any point.
        f"lateral offset: final {traj.final_lateral_offset_m:+.2f} m,"
        f" max |offset| {float(np.max(np.abs(lat))):.2f} m",
        # Line 4: whether a stop event or yield event occurred during the clip.
        f"events: stop={traj.stop_event} yield={traj.yield_event}",
    ]
    # Heading is undefined at near-zero speed; on a stationary clip the
    # accumulated number is pure noise and would mislead the generator.
    # Only report total heading change if the car actually moved a meaningful distance (>= 5 m).
    if dist >= 5.0:
        # Insert the heading-change line as the 4th line (index 3), between lateral offset and events.
        lines.insert(3, f"total heading change: {traj.total_heading_change_deg:+.1f} deg")
        # lateral_offset_m is y in the frozen t=0 heading frame, so on a
        # curving road it accumulates the road's geometry (tens of meters),
        # swamping in-lane maneuvers (~0.3 m). Verified 2026-08-05: 91 deg
        # turn -> 22 m "offset" on b7f37a71. Warn the generator off it.
        if abs(traj.total_heading_change_deg) > 15.0:
            lines.append(
                "warning: the road curves along this clip, so lateral offset"
                " accumulates road geometry -- it does NOT measure in-lane"
                " position; do not build lateral checks from it. Use speed"
                " magnitudes and event TIMING instead."
            )
    else:
        # Otherwise, add a note explaining that heading is meaningless here because the car barely moved.
        lines.append("note: ego is nearly stationary this clip; correct behavior is staying stopped/creeping")
    # Return the finished list of summary lines.
    return lines


# Build the full dossier text for one clip: obstacle tracks + ego trajectory
# summary + (optionally) the ground-truth reasoning annotation.
def build_dossier(
    scene: SceneObstacles,
    gt_traj: TrajectoryFeatures,
    gt_coc: str | None = None,
) -> str:
    """Render the dossier text handed to the generator."""
    # Start building the dossier as a list of text lines: a header with the clip ID,
    # a blank line, and a section header for the obstacle tracks.
    parts = [f"CLIP {scene.clip_id}", "", "OBSTACLE TRACKS (ego-relative, nearest first):"]
    # Add one line per obstacle track (already ranked nearest-first by summarize_tracks).
    for t in summarize_tracks(scene):
        parts.append(
            # Describe the track: its ID, class, when it was visible, and its closest approach.
            f"- track {t.track_id} [{t.label_class}] visible {t.t_enter_s:.1f}-{t.t_exit_s:.1f} s;"
            f" closest {t.closest_approach_m:.1f} m ({t.bearing_at_closest}) at t={t.t_closest_s:.1f} s"
        )
    # If there were more tracks than MAX_TRACKS, note how many were left out.
    if len(scene.tracks) > MAX_TRACKS:
        parts.append(f"({len(scene.tracks) - MAX_TRACKS} farther tracks omitted)")
    # Add a blank line and a section header for the expert ego trajectory.
    parts += ["", "EXPERT EGO TRAJECTORY (ground truth):"]
    # Add each ego-trajectory summary line, prefixed with "- " to match the bullet-point style above.
    parts += [f"- {line}" for line in ego_lines(gt_traj)]
    # If a ground-truth reasoning annotation string was provided, append it as its own section.
    if gt_coc:
        parts += ["", "GROUND-TRUTH REASONING ANNOTATION:", gt_coc.strip()]
    # Join all the lines together with newlines into the final dossier text.
    return "\n".join(parts)


# Convenience wrapper: build a TrajectoryFeatures object directly from raw
# (x, y) waypoints, without callers needing to know about extract_features's
# other required arguments.
def features_from_waypoints(waypoints: np.ndarray, hz: float, scene_id: str) -> TrajectoryFeatures:
    """Thin wrapper so callers outside pref_pairs don't repeat the boilerplate."""
    # Delegate to extract_features, filling in rollout_id=0 since this is always
    # treated as "the one and only" rollout in this context (not one of several sampled ones).
    return extract_features(waypoints, hz=hz, scene_id=scene_id, rollout_id=0)

# Clipgen process: 
# Here's the whole clipgen pipeline:
    # Inputs (per clip, from the dataset - 5 clips in the smoke run)
        # The obstacle tracks (where every labeled object was, over time)
        # The expert driver's actual trajectory (from the egomotion log)
        # The GT reasoning sentence ("Decelerate to maintain a safe distance")

    # Step 1: Build th edossier (dossier.py): reduces the scene to ~25 lines of text with real numbers: each obstacle's closest approach, when, and from which direction; what the expert did (speed profile, stop, lateral movement); the GT reasoning sentence. 

    # Step 2: Parse the GT reasoning into claims (coc_claim_parser.py, pure code).
    # The sentence becomes structured objects: PerceptualClaim(entity="lead_vehicle", ...), CommitmentClaim(maneuever="decelerate"), ...). 
    # These are what the reward functions willr eceive as claims. 

    # Step 3: Generate the reawrd function (gpt-4o)
    # One 3-turn conversation per clip
        # 1. "Here's the dossier, name this scene's decisive events"
        # 2. "Define what faithful vs unfaitfjhful behavhior looks like for those events, quantitatively"
        # 3. "Here's the claims/traj API and the exact GT claim objects, write def reward (claims, traj)" No prescribed formula: the model designs the scoring logic itself, knowing the gate's two numeric requirements (GT pair >= 0.7; every corruption >= 0.4 below).
        # 4. Sandbox check 
        # 5. The gate (gate.py, pure code, free - no API calls)
            # Build 5 test cases and run the function on each:
                # Positive: GT claims + GT trajectory -> must core >= 0.7 
                # 4 corruptions of that same pair: trajectory reversed, trajectory flattened (no reaction), claims gutted, commitments deleted (each must score >= 0.4 below the positive). 
                # Pass = the function rewards the right thing and stops rewarding it when either half is corrupted.

        # Step 6: Retry loop (up to 3 attempts per clip):
            # On failure, the exact violation plus the measured trajectory numbers for each case go back to gpt-4o in the same conversation: "revesred_traj scored 0.75, needs <= 0.35; measured facts ..."

        # Step 7: 
            # Report and peresist. Passing functions are saved to reward_fns/<clip>.py; everything (scores, sources, transcriptes, feedback API costs) goes into report.json, printed into the workload logs. Success bar >= 4/5 clips pass. 

        # Policy produces 12 rolouts for a scene, the validated reward function ranks them, the argmax rollout goes through the same step-5 perturbation battery, built from the rollout itself this time, only if its corruptions drop does the rollout get trustred. 

        # positives should clear 0.7 (exeuction credit firint on the real GT trajectory) and do reversed/flattened trajectories finally drop >= 0.4 instead of tying the positive. 

# tracks
# revsersed trajectory
# weighting, let the llm decide (sometimes there may not be much to perceive)? or sometimes the commitment may be brittle?
# obstacle.parquet

# track: obstacle.offline.parquet - the dataset's ground-truth object annotations. Every labeled object carries a numeric track_id, and the dossier surfaces them in its track lines: - track 32 [trailer] visible 0.0-19.8s; closest 8.2m (ahead) at t=17.5s 
# How many tracks: far more than the dossier shows:

# ┌───────────────────────────────────┬──────────────┬────────────────────────────────────────────────────────────────┐
# │               Clip                │ Total tracks │                          Composition                           │
# ├───────────────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────┤
# │ 01340cf8 (oversize load)          │ 183          │ 141 automobiles, 28 heavy trucks, 11 trailers, 2 riders, 1 bus │
# ├───────────────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────┤
# │ 40597645 (pedestrian)             │ 45           │ 42 automobiles, 3 persons                                      │
# ├───────────────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────┤
# │ b7f37a71 (construction cone)      │ 141          │ 115 automobiles, 13 persons, 5 protruding objects, 4 animals…  │
# ├───────────────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────┤
# │ eece4a2f (cyclist + stop sign)    │ 301          │ 229 automobiles, 50 persons, …                                 │
# ├───────────────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────┤
# │ fe20b8b9 (cyclist + lead vehicle) │ 15           │ 9 automobiles, 4 persons, 2 riders                             │
# └───────────────────────────────────┴──────────────┴────────────────────────────────────────────────────────────────┘

# global code as a reward parser
    # inspect the logs very clearly for this one. 

# slow down speed up
    # where does the minimum occur?
# revseed and global metrics not local ones?
# track from dossier. 
# lateral featurization bug?


# make updates to internvla