"""clip a8ad8f09-dac8-416f-808c-16b3a02e9102 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Decisive events: Pedestrian crossing (yield/decelerate).
    Trajectory thresholds: speed drop >= 1.15 m/s, min speed at t <= 3.0s.
    """

    # Initialize component scores
    comp = {
        "saw_pedestrian": 0.0,
        "yield_executed": 0.0
    }

    # Perceptual claim for pedestrian
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.1

    # Commitment claim for yielding/decelerating
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop and timing
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        min_speed_time = min_speed_time_idx * traj.dt_s

        # Graded factor for speed drop and timing
        if min_speed_time <= 3.0:
            comp["yield_executed"] = 0.6 * min(1.0, speed_drop / 2.3)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
