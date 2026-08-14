"""clip d4915d02-e9f8-49f5-9a88-158619ca749f - attempt 1/5 - gate PASS (pos 0.79, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive events:
    1. Traffic Officer Signal: Expect a mention of 'signal' and a commitment to 'decelerate'.
       Trajectory should show a speed drop of at least 1.35 m/s by around t=5.8 s.
    2. Proximity of Track 38 [Person]: Expect a mention of 'pedestrian' and a commitment to 'decelerate'.
       Trajectory should show a similar speed drop.
    3. Proximity of Track 56 [Automobile]: Expect a mention of 'vehicle_generic' and a commitment to 'decelerate'.
       Trajectory should show a similar speed drop.
    """

    # Initialize component scores
    scores = {
        "mention_signal": 0.0,
        "mention_pedestrian": 0.0,
        "mention_vehicle": 0.0,
        "decelerate_execution": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity == 'signal' for p in claims.perceptual):
        scores["mention_signal"] = 0.1

    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        scores["mention_pedestrian"] = 0.1

    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        scores["mention_vehicle"] = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor based on speed drop, with a floor at 1.35 m/s
        scores["decelerate_execution"] = 0.7 * min(1.0, speed_drop / 2.7)

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
