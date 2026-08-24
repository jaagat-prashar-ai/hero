"""clip e7b1c996-9522-4abd-a43d-b376797e56d5 - attempt 5/5 - gate PASS (pos 0.79, max pert 0.00, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scene e7b1c996-9522-4abd-a43d-b376797e56d5:
    - Stop at the stop sign: decelerate commitment, speed drop >= 0.0 m/s
    - Wait for cyclists: decelerate commitment, low speed maintenance
    """
    comp = {
        "stop_executed": 0.0,
        "wait_executed": 0.0,
    }

    # Trajectory analysis
    if traj.n_waypoints > 0:
        # Speed drop for stopping
        initial_speed = traj.speed_mps[0]
        min_speed = min(traj.speed_mps)
        speed_drop = initial_speed - min_speed

        # Stop execution: decelerate commitment and speed drop
        if any(c.speed_profile == 'decelerate' for c in claims.commitments):
            comp["stop_executed"] = 0.3 * min(1.0, speed_drop / 0.1)

        # Wait execution: low speed maintenance
        if any(c.speed_profile == 'decelerate' for c in claims.commitments):
            # Maintain low speed after initial deceleration
            low_speed_window = window(traj.speed_mps, traj.dt_s, 0.9, traj.n_waypoints * traj.dt_s)
            if len(low_speed_window) > 0:
                avg_low_speed = np.mean(low_speed_window)
                comp["wait_executed"] = 0.7 * min(1.0, (1.0 - avg_low_speed) / 1.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
