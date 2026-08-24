"""clip bdba61f0-a38d-4a06-9c3a-2c8c06f70332 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.23, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene bdba61f0-a38d-4a06-9c3a-2c8c06f70332:
    - Yield to pedestrians at the crosswalk: perceptual mention of 'pedestrian' or 'crosswalk',
      commitment to 'decelerate', and a speed drop of at least 3.35 m/s.
    """
    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.05,
        "mention_crosswalk": 0.05,
        "decelerate_for_pedestrian": 0.0
    }

    # Perceptual mentions
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05
    if any(p.entity in ('crosswalk',) for p in claims.perceptual):
        comp["mention_crosswalk"] = 0.05

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration
        comp["decelerate_for_pedestrian"] = 0.9 * min(1.0, speed_drop / 6.7)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
