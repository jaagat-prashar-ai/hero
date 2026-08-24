"""clip 3ccacc3e-1478-4a41-b78c-06160124af4a - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene with a decisive event of deceleration to maintain a safe distance from a lead vehicle.
    - Commitment: speed_profile='decelerate'
    - Trajectory: deceleration of at least 0.35 m/s, graded factor based on speed drop
    """
    commitment_trajectory_credit = 0.0

    # Commitment and Trajectory component: decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory component: graded deceleration
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 0.35:
            # Ensure the minimum speed occurs within the window
            min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
            if min_speed_time <= 3.0:  # Ensure the drop happens early in the window
                commitment_trajectory_credit = 0.7 * min(1.0, speed_drop / 0.7)

    return {
        "commitment_trajectory_decelerate": commitment_trajectory_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
