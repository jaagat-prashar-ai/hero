"""clip 56ff5406-cc81-4477-b384-cbdaf87698ba - attempt 2/3 - gate PASS (pos 0.80, max pert 0.40, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive events:
    - Gentle acceleration through the intersection.
    - Navigation around construction cones.
    
    Thresholds:
    - Speed increase from ~0.3 m/s to ~4.3 m/s, allowing for a range of 4.0 to 4.6 m/s.
    - Lateral offset within ±0.1 m of the GT's final offset of +0.40 m.
    - Total heading change between +5.0 and +9.0 degrees.
    """
    # Initialize component scores
    score = {
        "committed_to_accelerate_and_executed": 0.0,
        "committed_to_proceed_and_executed": 0.0,
        "executed_lateral_navigation": 0.0,
        "executed_heading_adjustment": 0.0
    }
    
    # Check commitment claims and trajectory execution
    if any(cc.maneuver == 'accelerate' and cc.speed_profile == 'accelerate' for cc in claims.commitments):
        # Check for speed increase
        if traj.n_waypoints > 0 and traj.final_speed_mps > traj.initial_speed_mps:
            score["committed_to_accelerate_and_executed"] = 0.4
    
    if any(cc.maneuver == 'proceed' for cc in claims.commitments):
        # Check for speed profile consistency
        if traj.n_waypoints > 0 and all(np.diff(traj.speed_mps) > 0):
            score["committed_to_proceed_and_executed"] = 0.2
    
    # Check trajectory execution
    if traj.n_waypoints > 0:
        # Lateral offset check
        if abs(traj.final_lateral_offset_m - 0.40) <= 0.1:
            score["executed_lateral_navigation"] = 0.2
        
        # Heading change check
        if 5.0 <= traj.total_heading_change_deg <= 9.0:
            score["executed_heading_adjustment"] = 0.2
    
    return score

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
