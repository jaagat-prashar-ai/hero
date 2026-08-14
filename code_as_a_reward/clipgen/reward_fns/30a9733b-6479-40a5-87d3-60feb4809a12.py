"""clip 30a9733b-6479-40a5-87d3-60feb4809a12 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on its reasoning and trajectory.
    Decisive events: Pedestrian presence requiring deceleration.
    Scene-derived thresholds: Speed drop of at least 0.3 m/s, with timing consideration.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    deceleration_executed = 0.0

    # Check for perceptual claims about pedestrians
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        perceptual_pedestrian = 0.1  # Small additive weight for mentioning pedestrians

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop within the trajectory
        initial_speed = traj.initial_speed_mps
        min_speed_after = traj.min_speed_mps
        speed_drop = initial_speed - min_speed_after

        # Find the time of minimum speed
        speed_window = window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)
        min_speed_idx = np.argmin(speed_window)
        min_speed_time = min_speed_idx * traj.dt_s

        # Graded factor for deceleration execution, considering timing
        if 3.0 <= min_speed_time <= 5.0:  # Timing window based on GT
            deceleration_executed = 0.6 * min(1.0, speed_drop / 0.6)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "deceleration_executed": deceleration_executed
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
