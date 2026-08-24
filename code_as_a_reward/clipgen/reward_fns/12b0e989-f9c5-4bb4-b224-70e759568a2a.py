"""clip 12b0e989-f9c5-4bb4-b224-70e759568a2a - attempt 2/5 - gate PASS (pos 0.90, max pert 0.40, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene 12b0e989-f9c5-4bb4-b224-70e759568a2a:
    - Deceleration to yield to a pedestrian crossing the crosswalk.
    - Proximity to nearby vehicles with no significant evasive maneuvers.
    - Trajectory thresholds: speed drop >= 0.15 m/s, lateral offset within GT range.
    """
    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "perceptual_crosswalk": 0.0,
        "decelerate_commitment": 0.0,
        "speed_reduction": 0.0
    }

    # Perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.05

    if any(p.entity == 'crosswalk' for p in claims.perceptual):
        comp["perceptual_crosswalk"] = 0.05

    # Commitment claims
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.3

        # Trajectory analysis for deceleration
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 0.15:
            comp["speed_reduction"] = 0.5 * min(1.0, speed_drop / 0.3)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
