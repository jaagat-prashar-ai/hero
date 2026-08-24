"""clip 17f8fddc-a313-4da1-9fa6-91158d13b9b0 - attempt 5/5 - gate PASS (pos 0.71, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Decisive event: Pedestrian crossing
    - Perceptual mention: pedestrian
    - Commitment: decelerate (speed_profile='decelerate')
    - Trajectory: speed drop of at least 0.5 m/s by around 3.1 seconds
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.05  # Minimal weight for mention-only credit

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop within the relevant time window
        speed_window = window(traj.speed_mps, traj.dt_s, 0.0, 3.1)
        if len(speed_window) > 0:
            initial_speed = speed_window[0]
            min_speed = np.min(speed_window)
            speed_drop = initial_speed - min_speed

            # Graded trajectory factor for deceleration
            trajectory_decelerate = 0.95 * min(1.0, speed_drop / 0.5)

            # Combine commitment and trajectory for deceleration
            commitment_decelerate = 0.70 * trajectory_decelerate

    # Return component scores
    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
