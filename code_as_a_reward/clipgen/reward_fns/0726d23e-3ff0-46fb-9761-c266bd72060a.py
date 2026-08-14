"""clip 0726d23e-3ff0-46fb-9761-c266bd72060a - attempt 2/5 - gate PASS (pos 0.95, max pert 0.25, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with deceleration for a curve and maintaining safe distance from construction zone.
    
    Decisive Events:
    1. Deceleration for the curve: Expect a speed drop of at least 4.6 m/s.
    2. Maintaining safe distance from construction zone: Expect careful navigation without specific lateral data.
    
    Trajectory thresholds are based on the GT's speed drop of 9.2 m/s.
    """
    # Initialize component scores
    comp = {
        "perceptual_curve": 0.0,
        "perceptual_construction_zone": 0.0,
        "decelerate_commitment": 0.0,
        "decelerate_execution": 0.0
    }
    
    # Perceptual claims
    if any(p.entity == 'curve' for p in claims.perceptual):
        comp["perceptual_curve"] = 0.05  # Small weight for mentioning the curve
    
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades') for p in claims.perceptual):
        comp["perceptual_construction_zone"] = 0.05  # Small weight for mentioning the construction zone
    
    # Commitment claims
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.2  # Weight for stating a deceleration commitment
    
    # Trajectory execution
    if traj.n_waypoints > 0:
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration execution
        comp["decelerate_execution"] = 0.7 * min(1.0, speed_drop / 9.2) if any(c.speed_profile == 'decelerate' for c in claims.commitments) else 0.0
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
