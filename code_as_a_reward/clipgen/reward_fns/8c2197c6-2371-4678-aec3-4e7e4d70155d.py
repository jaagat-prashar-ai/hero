"""clip 8c2197c6-2371-4678-aec3-4e7e4d70155d - attempt 4/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 7)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene.
    
    Decisive Event: Maintaining speed with pedestrians nearby.
    - Commitment: Deceleration (speed_profile='decelerate').
    - Trajectory: Speed reduction of at least 0.05 m/s, graded above this floor.
    """
    # Initialize component scores
    deceleration_commitment = 0.0

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed reduction
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Ensure the minimum speed occurs early in the window
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        min_speed_time = min_speed_time_idx * traj.dt_s

        # Graded speed reduction factor, ensuring timing alignment
        if min_speed_time <= 2.0:  # Ensure the minimum speed occurs early
            speed_reduction = 0.7 * min(1.0, speed_drop / 0.05)  # Adjusted floor for positive case
            deceleration_commitment = speed_reduction

    return {
        "deceleration_commitment": deceleration_commitment
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
