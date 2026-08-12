"""clip 9ddc1986-0495-4ee2-9255-1f772661ced6 - attempt 2/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 9)"""
def components(claims, traj):
    """Components for decisive events: 
    1. Lead Vehicle Deceleration and Stop
    Thresholds based on dossier: speed drop to ~0.3 m/s by 6.3s.
    """
    # Initialize component scores
    components = {
        "perceived_and_executed_deceleration": 0.0
    }
    
    # Check for perception and execution of deceleration
    perceived_lead_vehicle = any(
        claim.entity == 'lead_vehicle' for claim in claims.perceptual
    )
    committed_to_decelerate = any(
        commitment.maneuver == 'stop' and commitment.speed_profile == 'decelerate'
        for commitment in claims.commitments
    )
    
    if perceived_lead_vehicle and committed_to_decelerate:
        if traj.n_waypoints > 0:
            speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
            min_speed = np.min(speed_window) if len(speed_window) > 0 else float('inf')
            min_speed_time = np.argmin(speed_window) * traj.dt_s if len(speed_window) > 0 else float('inf')
            if min_speed <= 1.0 and 6.0 <= min_speed_time <= 6.4:
                components["perceived_and_executed_deceleration"] = 0.7

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
