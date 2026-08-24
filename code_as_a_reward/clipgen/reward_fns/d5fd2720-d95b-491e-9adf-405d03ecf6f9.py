"""clip d5fd2720-d95b-491e-9adf-405d03ecf6f9 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 10)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Pedestrian awareness: Mentions of 'pedestrian' or 'crosswalk'.
    - Deceleration commitment: Speed profile 'decelerate' with graded speed drop.
    - Deceleration execution: Graded speed drop in response to the commitment.
    Scene-derived thresholds:
    - Pedestrian speed drop: At least 1.75 m/s after t=3.6 s.
    """

    # Initialize component scores
    scores = {
        "pedestrian_mention": 0.0,
        "decelerate_commitment": 0.0,
        "decelerate_execution": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        scores["pedestrian_mention"] = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after_pedestrians = np.min(window(traj.speed_mps, traj.dt_s, 3.6, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after_pedestrians

        # Graded factor for deceleration commitment
        scores["decelerate_commitment"] = 0.45 * min(1.0, speed_drop / 3.5)

        # Graded factor for deceleration execution
        scores["decelerate_execution"] = 0.45 * min(1.0, speed_drop / 3.5)

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
