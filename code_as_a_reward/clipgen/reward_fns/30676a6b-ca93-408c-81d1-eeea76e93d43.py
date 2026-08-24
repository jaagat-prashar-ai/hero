"""clip 30676a6b-ca93-408c-81d1-eeea76e93d43 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene with gentle deceleration due to a protruding object and maintaining safe distance from nearby vehicles.
    
    Decisive Events:
    1. Gentle Deceleration due to a Protruding Object
       - Commitment: speed_profile='decelerate'
       - Trajectory: Speed drop >= 0.25 m/s, graded factor
    
    2. Maintaining Safe Distance from Nearby Automobiles
       - Commitment: lateral_maneuver='nudge', excluding direction='right'
       - Trajectory: Lateral offset >= 0.23 m, graded factor
    """
    
    # Initialize component scores
    comp = {
        "decelerate_executed": 0.0,
        "lateral_nudge_executed": 0.0
    }
    
    # Check for deceleration commitment and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 0.25:
            comp["decelerate_executed"] = 0.7 * min(1.0, speed_drop / 0.5)
    
    # Check for lateral nudge commitment and trajectory execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        lateral_offset = traj.final_lateral_offset_m
        if lateral_offset >= 0.23:
            comp["lateral_nudge_executed"] = 0.3 * min(1.0, lateral_offset / 0.47)
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
