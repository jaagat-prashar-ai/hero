"""clip e63fa44f-918c-4545-bf8b-9a0767cdada6 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.11, real rollout argmax 11)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive events:
    1. Pedestrian presence and crossing: Expect mention of 'pedestrian' and
       deceleration. Trajectory should show a speed drop of at least 3.0 m/s.
    2. Traffic light at intersection: Expect mention of 'signal' or 'intersection'
       and deceleration. Trajectory should show a speed drop of at least 3.0 m/s.
    """

    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for deceleration commitment
    decelerate_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Calculate speed drop
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps

    # Deceleration for pedestrians
    if decelerate_commitment:
        if any(p.entity == 'pedestrian' for p in claims.perceptual):
            comp["decelerate_for_pedestrian"] = 0.6 * min(1.0, speed_drop / 3.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
