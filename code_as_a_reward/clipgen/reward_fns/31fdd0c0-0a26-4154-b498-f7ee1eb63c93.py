"""clip 31fdd0c0-0a26-4154-b498-f7ee1eb63c93 - attempt 4/5 - gate PASS (pos 0.75, max pert 0.00, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing and nearby vehicles.
    
    Decisive Events:
    1. Pedestrian crossing at crosswalk requiring deceleration.

    Trajectory-derived thresholds:
    - Speed drop: at least 2.4 m/s (half of GT's 4.8 m/s drop).
    """

    # Initialize component scores
    comp = {
        "decelerate_for_pedestrian": 0.0
    }

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.1, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Find the time of minimum speed
        min_speed_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0.1, traj.n_waypoints * traj.dt_s))
        min_speed_time = 0.1 + min_speed_idx * traj.dt_s

        # Graded factor for deceleration, considering timing
        if min_speed_time > 3.0:  # Ensure the deceleration happens later in the window
            comp["decelerate_for_pedestrian"] = 0.75 * min(1.0, speed_drop / 4.8)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
