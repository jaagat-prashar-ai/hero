"""clip 231eac11-5dd5-49f1-aefa-58e7793479a4 - attempt 4/5 - gate PASS (pos 0.80, max pert 0.00, real rollout argmax 4)"""
def components(claims, traj):
    """
    Components for scoring a rollout's faithfulness to the scene:
    - Deceleration for pedestrian crossing: speed drop >= 0.7 m/s, occurring at t <= 2.1s
    """
    # Initialize component scores
    comp = {
        "decelerate_for_pedestrian": 0.0
    }

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Find the time of minimum speed
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
        # Graded factor for deceleration with timing condition
        if min_speed_time <= 2.1:
            comp["decelerate_for_pedestrian"] = 0.8 * min(1.0, speed_drop / 1.4)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
