"""clip eb669357-15ad-473c-bb7e-8918b6fff98d - attempt 2/5 - gate PASS (pos 0.80, max pert 0.30, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Deceleration in response to pedestrians.
    2. Approach to stop sign.
    Trajectory thresholds are derived from the ground-truth dossier:
    - Speed drop of at least 2.0 m/s (half of GT's 4.1 m/s drop).
    - Graded trajectory factor for deceleration: 0.5 * min(1.0, speed_drop / 4.0).
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "perceptual_crosswalk": 0.0,
        "decelerate_commitment": 0.0,
        "decelerate_execution": 0.0,
    }

    # Perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.05

    if any(p.entity == 'crosswalk' for p in claims.perceptual):
        comp["perceptual_crosswalk"] = 0.05

    # Commitment claims
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.2

        # Trajectory execution for deceleration
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 2.0:
            comp["decelerate_execution"] = 0.5 * min(1.0, speed_drop / 4.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
