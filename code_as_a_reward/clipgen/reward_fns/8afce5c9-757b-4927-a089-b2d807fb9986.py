"""clip 8afce5c9-757b-4927-a089-b2d807fb9986 - attempt 2/5 - gate PASS (pos 0.90, max pert 0.30, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scene with pedestrians crossing and nearby automobile.
    
    Decisive Events:
    1. Pedestrian crossing the road, requiring deceleration.
       - Perceptual mention: 'pedestrian'
       - Commitment: speed_profile='decelerate'
       - Trajectory: speed drop >= 2.5 m/s, graded factor
    2. Automobile on the left, requiring lateral adjustment.
       - Perceptual mention: 'vehicle_generic'
       - Commitment: lateral_maneuver='nudge'
       - Trajectory: lateral offset within ±0.14 m, graded factor
    """
    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "perceptual_vehicle": 0.0,
        "decelerate_execution": 0.0,
        "lateral_execution": 0.0
    }
    
    # Perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.1

    # Deceleration commitment and execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        comp["decelerate_execution"] = 0.6 * min(1.0, speed_drop / 4.9)

    # Lateral adjustment commitment and execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
        comp["lateral_execution"] = 0.2 * min(1.0, 0.14 / max_lateral_offset)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
