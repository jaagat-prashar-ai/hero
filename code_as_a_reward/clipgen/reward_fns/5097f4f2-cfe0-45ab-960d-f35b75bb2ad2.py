"""clip 5097f4f2-cfe0-45ab-960d-f35b75bb2ad2 - attempt 2/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of yielding to traffic in the intersection.
    Scene-derived thresholds:
    - Deceleration: Initial speed ~4.0 m/s to min speed ~0.5 m/s by ~4.7 s
    - Total speed drop: At least 3.0 m/s
    - Stop event: True within 6.4 s horizon
    - Commitment to yield with deceleration
    """

    # Initialize component scores
    commitment_and_execution_score = 0.0

    # Check commitment claims and trajectory execution together
    if any(claim.maneuver == 'yield' and claim.speed_profile == 'decelerate' for claim in claims.commitments):
        # Check trajectory execution
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        initial_speed = traj.initial_speed_mps
        min_speed = min(speed_window) if len(speed_window) > 0 else initial_speed
        speed_drop = initial_speed - min_speed

        # Ensure the speed drop occurs at an appropriate time
        min_speed_time_index = np.argmin(speed_window) if len(speed_window) > 0 else 0
        min_speed_time = min_speed_time_index * traj.dt_s

        if speed_drop >= 3.0 and min_speed_time >= 4.0:
            commitment_and_execution_score = 0.7

    return {
        "commitment_and_execution": commitment_and_execution_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
