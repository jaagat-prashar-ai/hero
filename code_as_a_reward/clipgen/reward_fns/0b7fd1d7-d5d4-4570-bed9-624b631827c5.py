"""clip 0b7fd1d7-d5d4-4570-bed9-624b631827c5 - attempt 3/5 - gate PASS (pos 0.95, max pert 0.53, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Pedestrian crossing: Decelerate to maintain a safe distance.
       - Perceptual mention: 'pedestrian'
       - Commitment: 'decelerate'
       - Trajectory: Speed drop of at least 2.5 m/s, graded
    """

    # Initialize component scores
    scores = {
        "mention_pedestrian": 0.05,
        "decelerate_for_pedestrian": 0.475,
    }

    # Perceptual mention checks
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        scores["mention_pedestrian"] = 0.05

    # Trajectory analysis
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    deceleration_factor = 0.9 * min(1.0, speed_drop / 5.1)

    # Commitment and trajectory checks for pedestrian
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        if speed_drop >= 2.5:
            scores["decelerate_for_pedestrian"] = deceleration_factor

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
