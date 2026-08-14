"""clip 033c9e61-c529-4e08-8b64-266db4edaf6e - attempt 4/5 - gate PASS (pos 0.99, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Strong deceleration for pedestrian crossing.
       - Perceptual mention: pedestrian-related entities.
       - Commitment: speed_profile='decelerate'.
       - Trajectory: speed drop of at least 3.9 m/s, graded.
    """
    # Initialize component scores
    comp = {
        "saw_pedestrian": 0.0,
        "decelerate_executed": 0.0,
    }

    # Check for pedestrian-related perceptual claims
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.1

    # Check for deceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 3.9:
            comp["decelerate_executed"] = 0.9 * min(1.0, speed_drop / 7.8)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
