"""clip 3a1dab1c-6c9e-4e84-9bea-5861c58b4fb5 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.23, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrians crossing.
    
    Decisive Events:
    1. Pedestrians crossing at the crosswalk: Expect mention of 'pedestrian' or 'crosswalk' and a deceleration commitment.
    
    Trajectory Expectations:
    - Speed drop of at least 1.4 m/s (half of the positive case drop of 2.8 m/s).
    - Graded factor for speed drop: 0.65 * min(1.0, speed_drop / 2.8).
    """

    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.05,
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        graded_speed_drop = 0.65 * min(1.0, speed_drop / 2.8)

        # Deceleration for pedestrians
        if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
            comp["decelerate_for_pedestrian"] = graded_speed_drop

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
