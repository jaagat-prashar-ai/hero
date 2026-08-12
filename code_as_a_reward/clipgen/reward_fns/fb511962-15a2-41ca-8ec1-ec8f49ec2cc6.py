"""clip fb511962-15a2-41ca-8ec1-ec8f49ec2cc6 - attempt 3/3 - gate PASS (pos 0.80, max pert 0.30, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive events:
    1. Stopping at the stop sign.
    
    Scene-derived thresholds:
    - Speed drop to 0.0 m/s within ~6.1s.
    - Stop event should occur within ±0.5s of the GT stop time.
    - Presence of perceptual claims for lead vehicle.
    - Commitment to stop.
    """
    # Initialize component scores
    scores = {
        "perceive_lead_vehicle": 0.0,
        "commit_stop": 0.0,
        "execute_stop": 0.0
    }
    
    # Check perceptual claims
    if any(claim.entity == 'lead_vehicle' for claim in claims.perceptual):
        scores["perceive_lead_vehicle"] = 0.1
    
    # Check commitment claims
    if any(claim.maneuver == 'stop' for claim in claims.commitments):
        scores["commit_stop"] = 0.2
    
    # Check trajectory execution for stopping with claim requirement
    if scores["commit_stop"] > 0.0:
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        if len(speed_window) > 0:
            min_speed = np.min(speed_window)
            stop_time = traj.dt_s * np.argmin(speed_window)
            if min_speed <= 0.1 and 5.6 <= stop_time <= 6.6:
                scores["execute_stop"] = 0.5  # Requires both claim and execution
    
    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
