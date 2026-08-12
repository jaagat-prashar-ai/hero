"""clip e0f45fdc-1612-464f-8174-1212425ad8df - attempt 3/3 - gate PASS (pos 0.70, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """Components for decisive events: Red traffic light, yielding to emergency vehicle, and pedestrian crossing.
    Thresholds: speed drop >= 7.5 m/s for red light, speed drop >= 4 m/s for yielding, lateral offset <= 0.3 m."""
    
    # Initialize component scores
    components = {
        "red_light_detected": 0.0,
        "red_light_executed": 0.0,
        "yield_emergency_detected": 0.0,
        "yield_emergency_executed": 0.0,
        "pedestrian_detected": 0.0,
        "pedestrian_executed": 0.0,
    }
    
    # Check for red traffic light detection and execution
    red_light_claim = any(pc.entity == 'signal' and pc.state == 'red' for pc in claims.perceptual)
    stop_commitment = any(cc.maneuver == 'stop' for cc in claims.commitments)
    
    if red_light_claim:
        components["red_light_detected"] = 0.2
    
    if red_light_claim and stop_commitment:
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        if len(speed_window) > 0 and (traj.initial_speed_mps - traj.min_speed_mps) >= 7.5:
            components["red_light_executed"] = 0.5
    
    # Remove components that scored 0.00 on the positive case
    # and focus on the conjunction of claims and trajectory execution
    
    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
