"""clip 733a71fb-5008-4260-b424-cbcd23f0fa87 - attempt 5/5 - gate PASS (pos 0.71, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 733a71fb-5008-4260-b424-cbcd23f0fa87:
    - Decisive event: Pedestrian presence and clearance
    - Perceptual entity: pedestrian
    - Commitment family: speed_profile='decelerate'
    - Trajectory expectation: speed drop of at least 1.8 m/s
    """
    # Initialize component scores
    perceptual_credit = 0.0
    commitment_credit = 0.0

    # Check for perceptual mention of pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_credit = 0.05  # Small additive weight for mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Ensure the minimum speed occurs at the expected time
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints)) * traj.dt_s
        if min_speed_time >= 6.0:  # Ensure the minimum speed is reached towards the end of the window
            # Graded trajectory factor for speed drop
            trajectory_credit = 0.95 * min(1.0, speed_drop / 3.6)
            # Combine commitment and trajectory credit
            commitment_credit = 0.7 * trajectory_credit

    # Return component contributions
    return {
        "perceptual_mention": perceptual_credit,
        "commitment_execution": commitment_credit
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
