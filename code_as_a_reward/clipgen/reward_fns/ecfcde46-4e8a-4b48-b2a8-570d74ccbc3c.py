"""clip ecfcde46-4e8a-4b48-b2a8-570d74ccbc3c - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 8)"""
def components(claims, traj):
    """Components for scene with strong deceleration due to lead vehicle.
    
    Decisive Events:
    1. Deceleration due to lead vehicle (heavy truck) ahead.
       - Perceptual entity family: {'lead_vehicle', 'vehicle_generic'}
       - Commitment family: speed_profile='decelerate'
       - Trajectory: speed drop >= 2.85 m/s, graded factor
    """
    # Initialize component scores
    comp = {
        "perceptual_lead_vehicle": 0.0,
        "decelerate_executed": 0.0
    }
    
    # Perceptual claims
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["perceptual_lead_vehicle"] = 0.05  # Reduced weight for mention-only credit

    # Commitment and trajectory for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 2.85:  # Half of the positive case's speed drop
            comp["decelerate_executed"] = 0.65 * min(1.0, speed_drop / 5.7)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
