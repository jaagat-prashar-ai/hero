"""clip e7301401-6f45-4240-bbd9-c7020dbcf75d - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for reward function based on decisive event: deceleration due to a closed gate ahead.
    - Commitment to 'decelerate' (speed_profile family) with trajectory showing a speed drop.
    """
    # Initialize component scores
    deceleration_commitment = 0.0

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop in the first 2.4 seconds
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 2.4)
        if len(speed_window) > 0:
            initial_speed = speed_window[0]
            min_speed = np.min(speed_window)
            speed_drop = initial_speed - min_speed
            # Graded trajectory factor for deceleration
            trajectory_deceleration = 0.7 * min(1.0, speed_drop / 0.2)
        
        # Combine commitment and trajectory check
        if trajectory_deceleration > 0:
            deceleration_commitment = trajectory_deceleration

    # Return component scores
    return {
        "deceleration_commitment": deceleration_commitment
    }

def reward(claims, traj):
    # Calculate total reward as the clamped sum of component scores
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
