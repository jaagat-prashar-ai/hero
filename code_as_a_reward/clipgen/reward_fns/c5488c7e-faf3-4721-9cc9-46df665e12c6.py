"""clip c5488c7e-faf3-4721-9cc9-46df665e12c6 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with gentle deceleration to maintain safe distance from pedestrians and oncoming traffic."""
    # Initialize component scores
    perceptual_pedestrian = 0.0
    deceleration_commitment = 0.0

    # Check for perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.05  # Small weight for mentioning pedestrians

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        # Adjusted to consider the timing of the minimum speed
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
        if min_speed_time >= 3.0:  # Ensure the deceleration happens later in the window
            deceleration_commitment = 0.65 * min(1.0, speed_drop / 2.4)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "deceleration_commitment": deceleration_commitment,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
