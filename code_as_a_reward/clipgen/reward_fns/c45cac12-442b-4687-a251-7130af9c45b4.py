"""clip c45cac12-442b-4687-a251-7130af9c45b4 - attempt 2/3 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 8)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of yielding to a pedestrian.
    The scene-derived thresholds are based on the expert's speed reduction and timing.
    """
    # Initialize component scores
    score_perceptual_claim = 0.0
    score_commitment_claim = 0.0
    score_trajectory_execution = 0.0

    # Check for perceptual claim: detecting the pedestrian
    for perceptual in claims.perceptual:
        if perceptual.entity == 'pedestrian' and perceptual.state == 'crossing':
            score_perceptual_claim = 0.2
            break

    # Check for commitment claim: yielding
    for commitment in claims.commitments:
        if commitment.maneuver == 'yield' and commitment.speed_profile == 'decelerate':
            score_commitment_claim = 0.2
            break

    # Check for trajectory execution: speed reduction
    if traj.n_waypoints > 0:
        # Extract speed series within the relevant time window
        speed_series = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        if len(speed_series) > 0:
            initial_speed = speed_series[0]
            min_speed = np.min(speed_series)
            final_speed = speed_series[-1]

            # Check if speed reduction is within acceptable bounds
            if initial_speed >= 3.4 and min_speed <= 1.8 and final_speed <= 3.1:
                # Check if the minimum speed occurs around the expected time
                min_speed_time = np.argmin(speed_series) * traj.dt_s
                if 5.2 <= min_speed_time <= 6.2:  # Adjusted timing window
                    # Conjunction of claims and trajectory execution
                    if score_perceptual_claim > 0 and score_commitment_claim > 0:
                        score_trajectory_execution = 0.6

    return {
        "perceptual_claim": score_perceptual_claim,
        "commitment_claim": score_commitment_claim,
        "trajectory_execution": score_trajectory_execution
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
