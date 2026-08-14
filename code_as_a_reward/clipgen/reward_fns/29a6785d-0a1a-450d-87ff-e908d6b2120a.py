"""clip 29a6785d-0a1a-450d-87ff-e908d6b2120a - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive events:
    1. Pedestrian crossing: Expect mention of 'pedestrian' and a deceleration commitment.
       Trajectory should show a speed drop of at least 1.8 m/s by t=5.9s.
    """

    # Initialize component scores
    comp = {
        "saw_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for perceptual claims
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.1

    # Check for commitment claims
    deceleration_claim = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Trajectory analysis
    if traj.n_waypoints > 0:
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Determine the time of minimum speed
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        min_speed_time = min_speed_time_idx * traj.dt_s

        # Check for deceleration commitment and trajectory execution
        if deceleration_claim and min_speed_time <= 5.9:
            comp["decelerate_for_pedestrian"] = 0.6 * min(1.0, speed_drop / 3.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
