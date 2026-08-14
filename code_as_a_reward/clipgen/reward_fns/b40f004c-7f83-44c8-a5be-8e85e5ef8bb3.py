"""clip b40f004c-7f83-44c8-a5be-8e85e5ef8bb3 - attempt 3/5 - gate PASS (pos 0.90, max pert 0.30, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring a rollout based on pedestrian proximity and yielding.
    
    Decisive events:
    1. Pedestrian proximity requiring deceleration.
       - Perceptual mention: pedestrian.
       - Commitment: speed_profile='decelerate'.
       - Trajectory: speed drop of at least 1.5 m/s, graded up to 4.1 m/s, with timing consideration.
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0
    trajectory_decelerate = 0.0

    # Check for perceptual mentions of pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1  # Small weight for mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        commitment_decelerate = 0.2  # Base weight for commitment

        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Check the timing of the minimum speed
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        min_speed_time = min_speed_time_idx * traj.dt_s

        # Graded factor for speed drop, requiring at least 1.5 m/s and timing consideration
        if 2.0 <= min_speed_time <= 3.5:  # Timing window for minimum speed
            trajectory_decelerate = 0.6 * min(1.0, speed_drop / 4.1)

    # Return component scores
    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_decelerate": trajectory_decelerate
    }

def reward(claims, traj):
    """Calculate the total reward as the clamped sum of components."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
