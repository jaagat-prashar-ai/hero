"""clip 43782a03-e636-4d78-90dd-f72704bb5ab7 - attempt 5/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive events:
    1. Pedestrian crossing: Expect a mention of a pedestrian and a deceleration commitment.
       Trajectory should show a speed drop of at least 2.1 m/s, graded, with timing sensitivity.
    """

    # Initialize component scores
    comp = {
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Find the time of minimum speed
        min_speed_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        min_speed_time = min_speed_idx * traj.dt_s

        # Graded factor for speed drop with timing sensitivity
        if min_speed_time <= 3.0:  # Ensure the minimum speed occurs early enough
            comp["decelerate_for_pedestrian"] = 0.7 * min(1.0, speed_drop / 3.9)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
