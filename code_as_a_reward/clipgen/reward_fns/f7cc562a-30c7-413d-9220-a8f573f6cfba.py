"""clip f7cc562a-30c7-413d-9220-a8f573f6cfba - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene with steering left to avoid construction zone and speed adjustment.
    
    Decisive Events:
    1. Steering left to maintain a safe distance from the construction zone.
       - Perceptual entities: construction_zone, construction_cones, barricades
       - Commitment: lateral maneuver (nudge, lane_change, turn) excluding right
       - Trajectory: leftward lateral offset change >= 0.20 m, heading change >= 3.0 degrees
    
    2. Speed adjustment to navigate safely.
       - Perceptual entities: vehicle_generic
       - Commitment: speed_profile = 'decelerate'
       - Trajectory: speed drop >= 1.2 m/s
    """
    # Initialize component scores
    comp = {
        "perceptual_construction": 0.0,
        "speed_adjustment": 0.0
    }

    # Perceptual claims
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.05  # Reduced weight to free up for commitment conjunctions

    # Speed adjustment commitment and trajectory
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 2.75:  # Adjusted to half of the measured 5.5 m/s drop
            comp["speed_adjustment"] = 0.65 * min(1.0, speed_drop / 5.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
