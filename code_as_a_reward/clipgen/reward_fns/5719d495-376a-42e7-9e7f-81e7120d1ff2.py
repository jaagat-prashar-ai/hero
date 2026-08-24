"""clip 5719d495-376a-42e7-9e7f-81e7120d1ff2 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene with road curvature preparation:
    - Commitment to decelerate (speed_profile='decelerate')
    - Trajectory shows significant speed adjustment (graded factor)
    """
    # Initialize component scores
    deceleration_commitment = 0.0

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop within the window
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after_idx = np.argmin(window(speed_series, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        min_speed_after = speed_series[min_speed_after_idx]
        speed_drop = initial_speed - min_speed_after

        # Ensure the speed drop occurs at the correct time
        min_speed_time = min_speed_after_idx * traj.dt_s
        if 3.0 <= min_speed_time <= 5.0:  # Adjusted timing window for speed drop
            # Graded factor for speed adjustment
            speed_adjustment = 0.7 * min(1.0, speed_drop / 3.85)  # Floor at half of 7.7 m/s drop

            # Combine commitment and trajectory for deceleration
            deceleration_commitment = speed_adjustment

    return {
        "deceleration_commitment": deceleration_commitment
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
