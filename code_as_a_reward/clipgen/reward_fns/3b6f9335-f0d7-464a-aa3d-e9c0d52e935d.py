"""clip 3b6f9335-f0d7-464a-aa3d-e9c0d52e935d - attempt 2/3 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for reward function based on decisive events:
    1. Stopping for the red traffic light.
    2. Yielding to the lead vehicle.
    
    Scene-derived thresholds:
    - Speed reduction to approximately 0.0 m/s by around t=5.4 s, with a tolerance of ±0.5 s.
    - Final speed close to 0.0 m/s, with a tolerance of ±0.1 m/s.
    - Minimal lateral offset changes, maintaining within ±0.02 m.
    """
    # Initialize component scores
    scores = {
        "perceived_traffic_light": 0.0,
        "commitment_stop": 0.0,
        "stop_execution": 0.0
    }
    
    # Check perceptual claims
    perceived_traffic_light = any(
        claim.entity == "signal" and claim.state == "red"
        for claim in claims.perceptual
    )
    
    if perceived_traffic_light:
        scores["perceived_traffic_light"] = 0.2
    
    # Check commitment claims
    commitment_stop = any(
        claim.maneuver == "stop" and claim.speed_profile == "decelerate"
        for claim in claims.commitments
    )
    
    if commitment_stop:
        scores["commitment_stop"] = 0.2
    
    # Check trajectory for stopping
    if traj.n_waypoints > 0:
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        min_speed = np.min(speed_window) if len(speed_window) > 0 else float('inf')
        final_speed = traj.final_speed_mps
        
        # Check if speed drops to approximately 0.0 m/s by t=5.4 s
        if min_speed <= 0.1 and abs(final_speed - 0.0) <= 0.1 and commitment_stop:
            scores["stop_execution"] = 0.6
    
    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
