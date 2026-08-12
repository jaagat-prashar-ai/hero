"""clip 02fd6a8f-db9c-423e-beb6-37729e7e6af2 - attempt 2/3 - gate PASS (pos 0.90, max pert 0.40, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for reward function based on decisive events:
    1. Stop for the stop sign held by the construction worker.
       - Commitment: 'stop' maneuver.
       - Trajectory: speed drop to approximately 0.0 m/s, minimal lateral offset.
    """

    # Initialize component scores
    committed_stop = 0.0
    executed_stop = 0.0

    # Check commitment claims for 'stop' maneuver
    if any(cc.maneuver == 'stop' for cc in claims.commitments):
        committed_stop = 0.3

    # Check trajectory for stopping behavior
    if traj.n_waypoints > 0:
        # Speed drop check
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        if len(speed_window) > 0:
            speed_drop = traj.initial_speed_mps - traj.min_speed_mps
            if speed_drop >= 0.15:  # Allow some tolerance from 0.2 m/s drop
                executed_stop += 0.2

        # Lateral offset check
        lateral_window = window(traj.lateral_offset_m, traj.dt_s, 0, 6.4)
        if len(lateral_window) > 0:
            max_lateral_offset = np.max(np.abs(lateral_window))
            if max_lateral_offset <= 0.02:  # Allow some tolerance from 0.01 m
                executed_stop += 0.2

    # Conjunction: require both commitment and execution for full credit
    if committed_stop > 0 and executed_stop > 0:
        executed_stop += 0.2  # Additional credit for matching claim and execution

    return {
        "committed_stop": committed_stop,
        "executed_stop": executed_stop
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
