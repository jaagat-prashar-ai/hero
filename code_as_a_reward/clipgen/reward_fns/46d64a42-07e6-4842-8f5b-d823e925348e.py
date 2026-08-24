"""clip 46d64a42-07e6-4842-8f5b-d823e925348e - attempt 3/5 - gate PASS (pos 0.90, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing and nearby vehicles.
    
    Decisive events:
    1. Yield to pedestrians crossing at the crosswalk.
    
    Trajectory thresholds:
    - Speed drop: at least 1.95 m/s (half of the positive's 3.9 m/s drop).
    """
    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "mention_vehicle": 0.0,
        "execute_yield": 0.0,
    }

    # Perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        comp["mention_vehicle"] = 0.1

    # Commitment to yield (decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Speed drop factor
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 1.95:
            comp["execute_yield"] = 0.8 * min(1.0, speed_drop / 3.9)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
