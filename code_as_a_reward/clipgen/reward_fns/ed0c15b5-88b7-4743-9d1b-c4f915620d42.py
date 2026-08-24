"""clip ed0c15b5-88b7-4743-9d1b-c4f915620d42 - attempt 4/5 - gate PASS (pos 0.75, max pert 0.02, real rollout argmax 2)"""
def components(claims, traj):
    """Components for the scene with pedestrians crossing.
    Decisive events:
    1. Pedestrian crossing: Expect deceleration with perceptual mention of pedestrians.
    Trajectory thresholds:
    - Speed drop: Minimum 0.95 m/s (half of GT's 1.9 m/s drop).
    - Timing: Minimum speed should occur after 3.0s to differentiate from reversed trajectory.
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0
    }

    # Check for perceptual mentions of pedestrians
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.02  # Minimal mention-only credit

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Find the time of minimum speed
        min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints))

        # Graded factor for speed drop with timing condition
        if min_speed_time > 3.0:  # Ensure minimum speed occurs after 3.0s
            comp["decelerate_for_pedestrian"] = 0.73 * min(1.0, speed_drop / 0.95)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
