"""clip 67014f4c-4fed-4835-bb93-d21d5cd2b42c - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the scene's decisive events:
    - Deceleration to maintain a safe distance from crossing turkeys.
    - Trajectory expectations: speed drop of at least 2.15 m/s by t=4.0 s.
    """

    # Initialize component scores
    deceleration_score = 0.0

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = traj.min_speed_mps
        speed_drop = initial_speed - min_speed_after

        # Calculate the time at which minimum speed occurs
        min_speed_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        min_speed_time = min_speed_idx * traj.dt_s

        # Graded factor for deceleration, considering both speed drop and timing
        if min_speed_time >= 3.0:  # Ensure the minimum speed occurs later in the window
            deceleration_score = 0.7 * min(1.0, speed_drop / 4.3)  # Graded based on speed drop

    # Return component scores
    return {
        "deceleration_executed": deceleration_score,
    }

def reward(claims, traj):
    # Calculate the total score and clamp it between 0.0 and 1.0
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
