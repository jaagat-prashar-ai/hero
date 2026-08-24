"""clip 938e5901-ec92-4ef7-a18a-554fe127defb - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on decisive events:
    1. Lead Vehicle and Merging Vehicle Proximity: Expect deceleration to maintain safe distance.
       - Perceptual mention: {'lead_vehicle', 'vehicle_generic'}
       - Commitment: speed_profile='decelerate'
       - Trajectory: Speed drop should occur around t=2.5s, with a floor at 0.3 m/s.
    """

    # Initialize component scores
    scores = {
        "perceptual_lead_vehicle": 0.05,
        "decelerate_execution": 0.0,
    }

    # Perceptual checks (mention-only credit)
    if any(p.entity in {'lead_vehicle', 'vehicle_generic'} for p in claims.perceptual):
        scores["perceptual_lead_vehicle"] = 0.05

    # Commitment and trajectory checks
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Speed drop should occur around t=2.5s
        min_speed = traj.min_speed_mps
        speed_drop = traj.initial_speed_mps - min_speed
        min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        if 1.5 <= min_speed_time <= 3.5:  # Adjusted time window for expected deceleration
            scores["decelerate_execution"] = 0.65 * min(1.0, speed_drop / 0.3)

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
