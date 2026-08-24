"""clip 7beb58eb-d4b0-4816-b72d-4ed879c1c859 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 9)"""
def components(claims, traj):
    """
    Components for scene 7beb58eb-d4b0-4816-b72d-4ed879c1c859:
    - Deceleration in response to pedestrians: speed drop of at least 0.4 m/s
    - Perceptual mention of pedestrians
    """

    # Initialize component scores
    comp = {
        "decelerate_for_pedestrians": 0.0,
        "mention_pedestrians": 0.0
    }

    # Check for perceptual mention of pedestrians
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["mention_pedestrians"] = 0.05

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Check if the minimum speed occurs early in the trajectory
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
        if min_speed_time <= 3.2:  # Ensure the drop happens early
            # Graded factor for deceleration
            comp["decelerate_for_pedestrians"] = 0.65 * min(1.0, speed_drop / 0.4)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
