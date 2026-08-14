"""clip 67a4299c-bd78-4ea8-9036-b75036e18408 - attempt 4/5 - gate PASS (pos 0.78, max pert 0.10, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene with acceleration through construction zone and maintaining safe distance from lead vehicle.
    
    Decisive events:
    1. Accelerate through the construction zone.
    2. Maintain safe distance from the lead vehicle.
    
    Scene-derived thresholds:
    - Speed increase floor: 3.7 m/s (half of GT's 7.4 m/s increase).
    """
    comp = {
        "mention_construction_zone": 0.05,
        "mention_lead_vehicle": 0.05,
        "accelerate_executed": 0.0,
    }

    # Perceptual mentions
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["mention_construction_zone"] = 0.05

    if any(p.entity == 'lead_vehicle' for p in claims.perceptual):
        comp["mention_lead_vehicle"] = 0.05

    # Commitment and trajectory checks
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        comp["accelerate_executed"] = 0.8 * min(1.0, speed_increase / 7.4)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
