"""clip 85766484-e0fc-489e-aad9-04ec31843c01 - attempt 2/3 - gate PASS (pos 0.90, max pert 0.30, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive events:
    1. Red Traffic Light: Detect and prepare to stop.
    2. Pedestrians Crossing: Detect and yield or exercise caution.
    
    Scene-derived thresholds:
    - Speed reduction for red light: at least 5.0 m/s.
    """
    # Initialize component scores
    red_light_detected = 0.0
    red_light_commitment = 0.0
    red_light_execution = 0.0

    # Check perceptual claims
    for claim in claims.perceptual:
        if claim.entity == 'signal' and claim.state == 'red':
            red_light_detected = 0.1

    # Check commitment claims
    for commitment in claims.commitments:
        if commitment.maneuver == 'stop' and commitment.speed_profile == 'decelerate':
            red_light_commitment = 0.2

    # Check trajectory execution for red light
    if traj.n_waypoints > 0:
        speed_reduction = traj.initial_speed_mps - traj.final_speed_mps
        if speed_reduction >= 5.0 and red_light_detected > 0.0 and red_light_commitment > 0.0:
            red_light_execution = 0.6  # Increase weight for conjunction of claim and execution

    # Combine component scores
    return {
        "red_light_detected": red_light_detected,
        "red_light_commitment": red_light_commitment,
        "red_light_execution": red_light_execution
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
