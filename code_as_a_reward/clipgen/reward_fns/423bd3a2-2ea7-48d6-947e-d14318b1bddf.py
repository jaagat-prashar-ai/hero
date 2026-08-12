"""clip 423bd3a2-2ea7-48d6-947e-d14318b1bddf - attempt 2/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of stopping for the red traffic light.
    Scene-derived thresholds are used for speed and timing checks.
    """
    # Initialize component scores
    committed_and_executed_stop = 0.0

    # Check for both commitment to stop and execution of stop
    if any(cc.maneuver == 'stop' and cc.speed_profile == 'decelerate' for cc in claims.commitments):
        if traj.n_waypoints > 0:
            # Extract speed over the trajectory window
            speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
            min_speed = np.min(speed_window) if len(speed_window) > 0 else float('inf')
            final_speed = traj.final_speed_mps

            # Check if the vehicle stops within the acceptable time range
            stop_time_window = window(traj.speed_mps, traj.dt_s, 4.5, 5.8)
            if min_speed <= 0.1 and final_speed <= 0.1 and len(stop_time_window) > 0 and np.min(stop_time_window) <= 0.1:
                committed_and_executed_stop = 0.7

    return {
        "committed_and_executed_stop": committed_and_executed_stop
    }

def reward(claims, traj):
    """
    Calculate the reward as the clamped sum of component contributions.
    """
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
