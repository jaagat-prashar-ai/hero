"""clip 79eb9733-1dfb-46fc-8561-78e6cac01a82 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.09, real rollout argmax 10)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing and automobiles on the right.
    
    Decisive Events:
    1. Pedestrian Crossing: Yield to pedestrian, decelerate by at least 4.0 m/s.
    2. Automobiles on the Right: Maintain lateral stability, no lateral maneuver towards right.
    
    Trajectory Thresholds:
    - Speed drop: At least 3.5 m/s (graded factor).
    """
    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0
    }

    # Check for perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration
        comp["decelerate_for_pedestrian"] = 0.65 * min(1.0, speed_drop / 3.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
