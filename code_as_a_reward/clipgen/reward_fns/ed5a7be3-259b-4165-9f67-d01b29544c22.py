"""clip ed5a7be3-259b-4165-9f67-d01b29544c22 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 6)"""
def components(claims, traj):
    """Components for scene ed5a7be3-259b-4165-9f67-d01b29544c22.
    
    Decisive events:
    1. Track 85 (Automobile Ahead): Requires deceleration to maintain a safe distance.
       - Perceptual mention: 'vehicle_generic'
       - Commitment: 'decelerate' (speed_profile)
       - Trajectory: Speed drop of at least 0.05 m/s by around 1.9 seconds.
    
    Trajectory thresholds are derived from the ground truth's in-window behavior.
    """

    # Initialize component scores
    perceptual_vehicle = 0.0
    deceleration_executed = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('vehicle_generic',) for p in claims.perceptual):
        perceptual_vehicle = 0.1

    # Check for deceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Ensure the minimum speed occurs early in the trajectory
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
        if min_speed_time <= 2.0:  # Ensure the minimum speed occurs early
            # Graded factor for deceleration execution
            deceleration_executed = 0.7 * min(1.0, speed_drop / 0.1)

    # Return component scores
    return {
        "perceptual_vehicle": perceptual_vehicle,
        "deceleration_executed": deceleration_executed
    }

def reward(claims, traj):
    # Calculate total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
