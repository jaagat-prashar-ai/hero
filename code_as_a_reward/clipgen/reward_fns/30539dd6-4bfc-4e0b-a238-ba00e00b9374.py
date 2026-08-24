"""clip 30539dd6-4bfc-4e0b-a238-ba00e00b9374 - attempt 2/5 - gate PASS (pos 0.90, max pert 0.10, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene with deceleration for construction zone.
    
    Decisive Event:
    1. Deceleration for Construction Zone: Adjust speed to maintain safe distance.
       - Perceptual: 'construction_zone', 'vehicle_ahead'
       - Commitment: Speed profile 'decelerate'
       - Trajectory: Speed drop >= 2.1 m/s, graded factor
    """
    # Initialize component scores
    comp = {
        "perceptual_construction": 0.1,
        "decelerate_maneuver": 0.0,
    }
    
    # Perceptual credit for construction zone
    if any(p.entity in ('work_zone', 'construction_cones') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.1
    
    # Deceleration commitment and trajectory check
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        speed_drop = initial_speed - final_speed
        comp["decelerate_maneuver"] = 0.8 * min(1.0, speed_drop / 4.2)
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
