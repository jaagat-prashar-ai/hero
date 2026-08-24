"""clip f013a049-b5f2-47a9-956d-a4c45dd86b02 - attempt 5/5 - gate PASS (pos 0.90, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for reward calculation based on decisive events:
    1. Pedestrian Crossing: Requires mention of 'pedestrian' and a 'decelerate' commitment.
    2. Trajectory should reflect minimal movement consistent with staying stopped or creeping.
    """

    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_commitment": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for decelerate commitment and corresponding trajectory behavior
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory analysis for stopping or creeping
        initial_speed = traj.initial_speed_mps
        min_speed_after = traj.min_speed_mps
        speed_drop = initial_speed - min_speed_after

        # Check the timing of the minimum speed
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        min_speed_time = min_speed_time_idx * traj.dt_s

        # Graded trajectory factor for stopping, conditioned on commitment and timing
        if speed_drop >= 0.5 and min_speed_time <= 2.0:
            comp["decelerate_commitment"] = 0.8 * min(1.0, speed_drop / 0.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
