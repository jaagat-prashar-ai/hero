"""clip 353f1108-91a1-40cc-a9d9-21dc83226f4d - attempt 1/5 - gate PASS (pos 0.70, max pert 0.20, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scene 353f1108-91a1-40cc-a9d9-21dc83226f4d:
    - Deceleration to yield to oncoming truck (track 95).
    - Thresholds: speed drop >= 3.8 m/s (half of GT's 7.6 m/s drop).
    - Perceptual mention of 'vehicle_generic', 'oncoming_traffic', or 'intersection'.
    """

    # Initialize component scores
    comp = {
        "perceptual_mention": 0.0,
        "deceleration_commitment": 0.0,
        "deceleration_execution": 0.0
    }

    # Perceptual mention credit
    if any(p.entity in ('vehicle_generic', 'oncoming_traffic', 'intersection') for p in claims.perceptual):
        comp["perceptual_mention"] = 0.1

    # Deceleration commitment credit
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["deceleration_commitment"] = 0.2

        # Trajectory execution credit for deceleration
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded factor for speed drop
        if speed_drop >= 3.8:  # Half of the GT's 7.6 m/s drop
            comp["deceleration_execution"] = 0.5 * min(1.0, speed_drop / 7.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
