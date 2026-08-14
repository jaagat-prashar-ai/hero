"""clip cb01e385-6a7b-4563-9ec6-9aed407cff48 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Pedestrian crossing: Deceleration to maintain safe distance.
       - Perceptual: 'pedestrian'
       - Commitment: 'decelerate'
       - Trajectory: Speed drop of at least 0.95 m/s within the first 2 seconds.
    """

    # Initialize component scores
    scores = {
        "saw_pedestrian": 0.0,
        "decelerate_executed": 0.0,
    }

    # Perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        scores["saw_pedestrian"] = 0.1

    # Commitment claims
    decelerate_claimed = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Trajectory analysis
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
    min_speed_time = min_speed_time_idx * traj.dt_s

    # Deceleration execution
    if decelerate_claimed and min_speed_time <= 2.0:
        scores["decelerate_executed"] = 0.6 * min(1.0, speed_drop / 1.9)

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
