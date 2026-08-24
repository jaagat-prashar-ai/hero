"""clip 29ce7800-de4d-4de1-92f8-4580aa48487e - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on:
    1. Deceleration in response to pedestrians.
    2. Mention of pedestrians.
    Thresholds derived from scene: speed drop >= 0.7 m/s.
    """

    # Initialize component scores
    components = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for perceptual mention of pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        components["mention_pedestrian"] = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Ensure the minimum speed occurs later in the window
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints))
        min_speed_time = min_speed_time_idx * traj.dt_s
        # Graded factor for deceleration, considering timing
        if min_speed_time > 3.0:  # Ensure the minimum speed occurs after t=3.0s
            components["decelerate_for_pedestrian"] = 0.6 * min(1.0, speed_drop / 3.1)

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
