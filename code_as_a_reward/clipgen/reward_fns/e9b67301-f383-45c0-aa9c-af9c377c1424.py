"""clip e9b67301-f383-45c0-aa9c-af9c377c1424 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the scene's decisive events:
    1. Pedestrian crossing at the crosswalk: Expect a deceleration commitment
       and mention of pedestrian or crosswalk. Trajectory should show a speed
       drop of at least 3.85 m/s (half of the positive case's 7.7 m/s drop).
    """

    # Initialize component scores
    comp = {
        "saw_pedestrian": 0.0,
        "decelerate_executed": 0.0,
    }

    # Check for perceptual claims
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration, conditioned on both claim and trajectory
        if speed_drop >= 3.85:
            comp["decelerate_executed"] = 0.9 * min(1.0, speed_drop / 7.7)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
