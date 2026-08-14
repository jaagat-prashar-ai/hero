"""clip 14daf39e-1c72-4fc6-9eb8-d2066c2c5e84 - attempt 2/5 - gate PASS (pos 0.90, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on decisive events:
    1. Yielding to the pedestrian: Expect a mention of 'pedestrian' and a
       commitment to 'decelerate'. Trajectory should show a speed reduction
       of at least 1.35 m/s, graded with a factor of 0.5 * min(1.0, drop / 2.7).
    2. Maintaining lane position: Minimal weight as it does not involve a
       specific perceptual or commitment claim. Lateral offset should remain
       within 0.15 m.
    """
    # Initialize component scores
    scores = {
        "mention_pedestrian": 0.0,
        "decelerate_commitment": 0.0,
        "speed_reduction": 0.0,
        "lateral_stability": 0.0
    }

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        scores["mention_pedestrian"] = 0.05  # Reduced weight

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed reduction
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        if speed_drop >= 1.35:
            scores["decelerate_commitment"] = 0.2
            scores["speed_reduction"] = 0.5 * min(1.0, speed_drop / 2.7)

    # Check for lateral stability
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
        scores["lateral_stability"] = 0.15 * min(1.0, 0.15 / max_lateral_offset)  # Reduced weight

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
