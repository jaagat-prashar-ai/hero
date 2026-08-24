"""clip e9e791dd-5308-48b1-ab8c-3835e011de55 - attempt 2/5 - gate PASS (pos 0.90, max pert 0.00, real rollout argmax 8)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    1. Commitment to decelerate with matching trajectory execution.
    
    Scene-derived thresholds:
    - Speed reduction: at least 2.1 m/s (half of GT drop of 4.2 m/s).
    - Commitment family: speed_profile='decelerate'.
    """
    # Initialize component scores
    commitment_and_trajectory_decelerate = 0.0

    # Check for commitment to decelerate and matching trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed reduction over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 2.1:  # Half of the positive drop of 4.2 m/s
            # Ensure the minimum speed occurs at the correct time
            min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
            if 2.0 <= min_speed_time <= 4.0:  # Allow some flexibility around the observed minimum time
                commitment_and_trajectory_decelerate = 0.9 * min(1.0, speed_drop / 4.2)

    # Return the component contributions
    return {
        "commitment_and_trajectory_decelerate": commitment_and_trajectory_decelerate,
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
