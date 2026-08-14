"""clip 027c10f0-ca30-414a-8fdd-d4200b736b9f - attempt 3/5 - gate PASS (pos 0.70, max pert 0.07, real rollout argmax 3)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the decisive events:
    1. Stopping for the red traffic light and pedestrian.
       - Perceptual mention: 'signal', 'pedestrian'
       - Commitment: 'decelerate' family
       - Trajectory: Speed drop of at least 2.0 m/s, ideally reaching 0.3 m/s by t=5.7s
    """

    # Initialize component scores
    perceptual_signal_pedestrian = 0.0
    stop_executed = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('signal', 'pedestrian') for p in claims.perceptual):
        perceptual_signal_pedestrian = 0.05  # Reduced weight for mention-only

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded factor for speed drop, adjusted for timing
        if traj.n_waypoints > 0:
            min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
            if min_speed_time <= 5.7:
                stop_executed = 0.65 * min(1.0, speed_drop / 4.0)

    # Return the component contributions
    return {
        "perceptual_signal_pedestrian": perceptual_signal_pedestrian,
        "stop_executed": stop_executed,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
