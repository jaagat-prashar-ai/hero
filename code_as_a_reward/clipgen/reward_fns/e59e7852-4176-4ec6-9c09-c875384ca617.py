"""clip e59e7852-4176-4ec6-9c09-c875384ca617 - attempt 4/5 - gate PASS (pos 0.84, max pert 0.00, real rollout argmax 11)"""
def components(claims, traj):
    """Components for the traffic light stop scene:
    - Commitment to decelerate with matching trajectory execution
    - Trajectory speed reduction of at least 0.05 m/s early in the window
    """
    comp = {
        "commitment_decelerate": 0.0,
        "trajectory_decelerate": 0.0
    }

    # Commitment to decelerate with matching trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Ensure the speed drop occurs early in the window
        min_speed_time = traj.dt_s * np.argmin(window(speed_series, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        
        if speed_drop >= 0.05 and min_speed_time <= 1.0:  # Half of the positive's 0.1 m/s drop, early in the window
            comp["commitment_decelerate"] = 0.3
            comp["trajectory_decelerate"] = 0.7 * min(1.0, speed_drop / 0.1)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
