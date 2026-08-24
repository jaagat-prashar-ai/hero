"""clip 3f88f1e5-3b59-4b1b-8e61-ee2e74d4e824 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 3f88f1e5-3b59-4b1b-8e61-ee2e74d4e824:
    - Maintain safe distance from vehicle ahead (decelerate commitment, speed drop >= 1.75 m/s)
    - Perceptual mention of vehicle_generic
    """
    comp = {
        "perceptual_vehicle": 0.0,
        "decelerate_executed": 0.0
    }
    
    # Perceptual mention of vehicle_generic
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.1

    # Decelerate commitment and execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 1.75:
            comp["decelerate_executed"] = 0.9 * min(1.0, speed_drop / 3.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
