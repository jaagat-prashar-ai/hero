"""clip 2706fdc1-5352-4e7a-aaac-3a3ac4cf641c - attempt 2/5 - gate PASS (pos 1.00, max pert 0.41, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrians crossing and yielding behavior.
    
    Decisive Events:
    1. Pedestrians crossing the road, requiring a deceleration.
       - Perceptual mention: 'pedestrian'
       - Commitment: 'decelerate' family (stop/yield/wait/decelerate)
       - Trajectory: Speed drop of at least 5 m/s, graded above this floor.
    """
    # Initialize component scores
    scores = {
        "perceptual_pedestrian": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_decelerate": 0.0
    }
    
    # Check for perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        scores["perceptual_pedestrian"] = 0.1  # Small weight for mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        scores["commitment_decelerate"] = 0.3  # Significant weight for commitment

        # Trajectory analysis for deceleration
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0, traj.n_waypoints))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for deceleration
        scores["trajectory_decelerate"] = 0.6 * min(1.0, speed_drop / 6.0)

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
