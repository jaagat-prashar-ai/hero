"""clip 4866dbaf-ad04-43d1-8bd6-9492d615da4e - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 7)"""
def components(claims, traj):
    """
    Decisive Events:
    1. Stop Sign and Pedestrian Crossing: Expect deceleration with a mention of 'pedestrian'.
       - Trajectory: Speed drop of at least 1.65 m/s by t=2.8 s.
    """
    # Initialize component scores
    comp = {
        "decelerate_executed": 0.0
    }

    # Commitment and trajectory check for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration with timing consideration
        if np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s <= 2.8:
            comp["decelerate_executed"] = 0.7 * min(1.0, speed_drop / 1.65)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
