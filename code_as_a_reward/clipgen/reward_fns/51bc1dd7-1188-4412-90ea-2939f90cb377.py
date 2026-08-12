"""clip 51bc1dd7-1188-4412-90ea-2939f90cb377 - attempt 2/3 - gate PASS (pos 0.70, max pert 0.20, real rollout argmax 9)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive events of
    navigating through traffic barrels with cautious speed control.
    """

    # Initialize component scores
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check for commitment to stop or creep and link it to trajectory execution
    if any(claim.maneuver in ['stop', 'creep'] for claim in claims.commitments):
        # Evaluate trajectory for speed and stop event
        if traj.n_waypoints > 0:
            # Speed should start near 0 and remain low, with a slight increase
            speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
            if len(speed_window) > 0:
                initial_speed = speed_window[0]
                final_speed = speed_window[-1]
                if 0.0 <= initial_speed <= 0.5 and 0.0 <= final_speed <= 2.0 and traj.stop_event:
                    commitment_score = 0.5

    # Evaluate trajectory for lateral offset
    if traj.n_waypoints > 0:
        lateral_window = window(traj.lateral_offset_m, traj.dt_s, 0, 6.4)
        if len(lateral_window) > 0 and np.all(np.abs(lateral_window) <= 0.1):
            trajectory_score = 0.2

    return {
        "commitment_claims": commitment_score,
        "trajectory_execution": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
