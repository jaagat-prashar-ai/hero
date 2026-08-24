"""clip 451cee08-6c55-480f-9a37-0dd5032323b6 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 9)"""
def components(claims, traj):
    """Components for scene with construction vehicle and intersection:
    - Acceleration after intersection: speed increase >= 0.85 m/s at correct timing
    - Perceptual mentions: vehicle_generic, intersection
    """

    # Initialize component scores
    comp = {
        "accelerate_after_intersection": 0.0,
        "mention_vehicle": 0.05,
        "mention_intersection": 0.05,
    }

    # Perceptual mentions
    if any(p.entity in ('vehicle_generic', 'stopped_vehicle') for p in claims.perceptual):
        comp["mention_vehicle"] = 0.05

    if any(p.entity == 'intersection' for p in claims.perceptual):
        comp["mention_intersection"] = 0.05

    # Acceleration after intersection
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        speed_increase = traj.final_speed_mps - traj.min_speed_mps
        if speed_increase >= 0.85:
            comp["accelerate_after_intersection"] = 0.9 * min(1.0, speed_increase / 4.8)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
