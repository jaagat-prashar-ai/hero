"""clip 0a876511-ef76-4b42-ba6c-530852cdb7d5 - attempt 1/5 - gate PASS (pos 0.90, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Traffic Light Stop: Expect mention of 'signal' and a deceleration commitment.
       Trajectory should show a speed drop of at least 4.7 m/s.
    2. Yield to Pedestrian: Expect mention of 'pedestrian' or 'crosswalk' and a deceleration commitment.
       Trajectory should contribute to the overall deceleration.
    """
    # Initialize component scores
    comp = {
        "mention_signal": 0.0,
        "mention_pedestrian": 0.0,
        "decelerate_for_signal": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity in ('signal',) for p in claims.perceptual):
        comp["mention_signal"] = 0.1

    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for deceleration commitment
    has_decelerate_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Calculate speed drop
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps

    # Deceleration for traffic light
    if has_decelerate_commitment:
        comp["decelerate_for_signal"] = 0.4 * min(1.0, speed_drop / 6.0)

    # Deceleration for pedestrian
    if has_decelerate_commitment:
        comp["decelerate_for_pedestrian"] = 0.4 * min(1.0, speed_drop / 6.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
