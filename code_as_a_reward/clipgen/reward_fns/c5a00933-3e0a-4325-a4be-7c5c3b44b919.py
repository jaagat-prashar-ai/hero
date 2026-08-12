"""clip c5a00933-3e0a-4325-a4be-7c5c3b44b919 - attempt 3/3 - gate PASS (pos 1.00, max pert 0.30, real rollout argmax 11)"""
def components(claims, traj):
    """Components for gentle deceleration due to upcoming temporary traffic delineators."""
    # Initialize component scores
    trajectory_execution_score = 0.0
    combined_score = 0.0

    # Check for the presence of the commitment to decelerate
    has_decelerate_commitment = any(
        commitment.maneuver == 'decelerate' and commitment.speed_profile == 'decelerate'
        for commitment in claims.commitments
    )

    # Check trajectory for gentle deceleration
    if traj.n_waypoints > 0:
        # Calculate speed drop over the trajectory
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        speed_drop = initial_speed - final_speed

        # Check if the speed drop is significant and occurs towards the end of the trajectory
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        if len(speed_window) > 0:
            min_speed = np.min(speed_window)
            min_speed_time = np.argmin(speed_window) * traj.dt_s
            if speed_drop >= 5.0 and final_speed <= 8.9 and min_speed_time >= 5.0:
                trajectory_execution_score = 0.3

    # Combined score requires both a commitment claim and matching trajectory execution
    if has_decelerate_commitment and trajectory_execution_score > 0:
        combined_score = 0.7

    # Total score is the sum of the components
    return {
        "trajectory_execution": trajectory_execution_score,
        "combined_claim_and_execution": combined_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
