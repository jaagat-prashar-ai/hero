"""clip b12278b9-d550-41ff-97db-ae9d41855079 - attempt 3/3 - gate PASS (pos 1.00, max pert 0.60, real rollout argmax 4)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the decisive event of yielding to a pedestrian.
    Scene-derived thresholds:
    - Speed reduction: significant drop from initial speed (8.2 m/s) to around 1.7-2.5 m/s by t=5.4 s.
    - Perceptual claim: detection of a pedestrian crossing.
    - Commitment claim: commitment to yield.
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check for perceptual claim about pedestrian crossing
    perceptual_claim_present = any(
        claim.entity == 'pedestrian' and claim.state == 'crossing'
        for claim in claims.perceptual
    )

    # Check for commitment to yield
    commitment_claim_present = any(
        claim.maneuver == 'yield' and claim.speed_profile == 'decelerate'
        for claim in claims.commitments
    )

    # Check trajectory for significant speed reduction within the expected timeframe
    if traj.n_waypoints > 0:
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Ensure speed drop is significant and occurs within the expected timeframe
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        if len(speed_window) > 0 and np.min(speed_window) <= 2.5:
            min_speed_time_index = np.argmin(speed_window)
            min_speed_time = min_speed_time_index * traj.dt_s
            if min_speed_time >= 5.0 and min_speed_time <= 6.4:
                trajectory_score = 0.6

    # Combine perceptual and commitment claims with trajectory execution
    if perceptual_claim_present and commitment_claim_present and trajectory_score > 0:
        perceptual_score = 0.2
        commitment_score = 0.2

    return {
        "perceptual_claim": perceptual_score,
        "commitment_claim": commitment_score,
        "trajectory_execution": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
