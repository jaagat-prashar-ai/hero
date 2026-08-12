"""clip 11820df1-8a6e-438b-a029-2f5295dcd9b8 - attempt 3/3 - gate PASS (pos 0.80, max pert 0.40, real rollout argmax 4)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "perceived_lead_vehicle": 0.0,
        "committed_keep_distance": 0.0,
        "executed_speed_reduction": 0.0
    }
    
    # Check for perception of the lead vehicle
    if any(pc.entity == 'lead_vehicle' for pc in claims.perceptual):
        scores["perceived_lead_vehicle"] = 0.2

    # Check for commitment to keep distance
    has_commitment = any(cc.maneuver == 'keep_distance' for cc in claims.commitments)
    if has_commitment:
        scores["committed_keep_distance"] = 0.2

    # Check for speed reduction execution with timing consideration
    speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
    if len(speed_window) > 0:
        initial_speed = traj.initial_speed_mps
        min_speed = min(speed_window)
        speed_drop = initial_speed - min_speed
        min_speed_time = traj.dt_s * np.argmin(speed_window)
        if has_commitment and 1.5 <= speed_drop <= 2.5 and 3.0 <= min_speed_time <= 4.0:
            scores["executed_speed_reduction"] = 0.4  # Requires both claim and execution

    return scores

def reward(claims, traj):
    """Decisive events: Maintain safe distance from lead vehicle.
    Scene-derived thresholds: Speed drop 1.5-2.5 m/s, min speed time 3.0-4.0s."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
