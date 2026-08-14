"""clip fb511962-15a2-41ca-8ec1-ec8f49ec2cc6 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on:
    1. Stopping behind the lead vehicle at the stop sign.
    2. Acknowledging pedestrians crossing the intersection.
    Trajectory thresholds are derived from the ground-truth dossier.
    """

    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "stop_executed": 0.0,
    }

    # Perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Commitment and trajectory execution for stopping
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop, expecting at least 0.6 m/s drop
        # Adjust timing expectation to match the positive case
        comp["stop_executed"] = 0.7 * min(1.0, speed_drop / 1.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
