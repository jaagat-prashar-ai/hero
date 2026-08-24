"""clip 865d4704-7bc4-4a86-9272-5b12a47f1146 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.30, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene with pedestrians crossing and no immediate stop.
    
    Decisive Events:
    1. Pedestrians crossing the road at a crosswalk.
       - Perceptual mention: 'pedestrian'
       - Commitment: 'decelerate' (stop/yield/wait/decelerate)
       - Trajectory: Speed drop of at least 1.5 m/s, graded factor.
    
    Thresholds:
    - Speed drop floor: 1.5 m/s
    - Graded speed drop factor: 0.6 * min(1.0, speed_drop / 2.9)
    """

    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.05,  # Reduced weight for mention-only
        "decelerate_executed": 0.0
    }

    # Check for perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 2.1, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop
        comp["decelerate_executed"] = 0.65 * min(1.0, speed_drop / 2.9)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
