"""clip c375e9a6-a9bf-4ab2-825e-6735da236b6f - attempt 2/5 - gate PASS (pos 1.00, max pert 0.00, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene c375e9a6-a9bf-4ab2-825e-6735da236b6f:
    - Decelerate to maintain a safe distance from nearby vehicles.
    - Graded trajectory factor based on readiness to decelerate.
    """
    commitment_weight = 0.7
    trajectory_weight = 0.3

    # Commitment to decelerate
    committed_to_decelerate = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Trajectory readiness to decelerate
    speed_series = np.array(traj.speed_mps)
    initial_speed = traj.initial_speed_mps
    min_speed_after = np.min(window(speed_series, traj.dt_s, 0.1, traj.n_waypoints * traj.dt_s))
    speed_drop = initial_speed - min_speed_after
    trajectory_score = 0.0

    if committed_to_decelerate:
        # Graded factor based on speed drop, with a generous floor
        trajectory_score = trajectory_weight * min(1.0, speed_drop / 2.7)

    # Combine commitment and trajectory for a real conjunction
    commitment_score = commitment_weight if committed_to_decelerate and speed_drop > 0 else 0.0

    return {
        "commitment_to_decelerate": commitment_score,
        "trajectory_readiness": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
