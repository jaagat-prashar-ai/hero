"""clip 4978d5d9-de81-4338-9565-d1590004046d - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene 4978d5d9-de81-4338-9565-d1590004046d:
    - Approaching the roundabout: expect deceleration and mention of roundabout/intersection.
    Trajectory thresholds: speed drop >= 1.85 m/s (half of the positive's 3.7 m/s drop).
    """

    # Initialize component scores
    perceptual_score = 0.0
    slowing_score = 0.0

    # Perceptual mention of roundabout or intersection
    if any(p.entity in ('roundabout', 'intersection') for p in claims.perceptual):
        perceptual_score = 0.1

    # Check for deceleration commitment and corresponding trajectory
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 1.85:  # Half of the positive's 3.7 m/s drop
            slowing_score = 0.6 * min(1.0, speed_drop / 3.7)

    return {
        "perceptual_mention": perceptual_score,
        "slowing_executed": slowing_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
