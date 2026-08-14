"""clip 5fa56c61-e8ed-4c0b-95dd-3e15c40d9f78 - attempt 2/5 - gate PASS (pos 0.99, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """Decisive events: (1) Deceleration to maintain safe distance from garbage truck.
    Scene-derived thresholds: speed drop >= 1.5 m/s. Perceptual entities: vehicle_generic."""
    
    # Initialize component scores
    comp = {
        "perceptual_vehicle": 0.0,
        "decelerate_execution": 0.0
    }
    
    # Perceptual claims
    if any(p.entity in ('vehicle_generic', 'lead_vehicle') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.05
    
    # Commitment claims and trajectory execution
    # Deceleration to maintain safe distance
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Adjust the graded factor to cover the weight of the removed component
        comp["decelerate_execution"] = 0.95 * min(1.0, speed_drop / 3.0)
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
