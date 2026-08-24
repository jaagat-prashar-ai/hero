"""clip 1ce86e13-03cf-4cb5-99b2-a65400650117 - attempt 5/5 - gate PASS (pos 1.00, max pert 0.00, real rollout argmax 8)"""
def components(claims, traj):
    """Components for scene 1ce86e13-03cf-4cb5-99b2-a65400650117:
    - Deceleration in response to turkeys crossing the road.
    - Thresholds: speed drop >= 3.0 m/s, trajectory graded on drop.
    """
    commitment_trajectory_credit = 0.0

    # Check for deceleration commitment and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Ensure the trajectory execution is conditioned on the commitment and timing
        if speed_drop >= 3.0:
            # Check the timing of the minimum speed
            min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
            if 3.0 <= min_speed_time <= 4.0:  # Ensure the minimum speed occurs within a reasonable time window
                commitment_trajectory_credit = 1.0 * min(1.0, speed_drop / 6.0)

    return {
        "commitment_trajectory": commitment_trajectory_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
