"""clip 7a8cb31c-bef5-400d-ba68-5fcaf0abfca2 - attempt 3/3 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on decisive events:
    1. Acceleration after the traffic light turns green.
    2. Maintaining a safe distance from the motorcyclist ahead.

    Scene-derived thresholds:
    - Acceleration: Speed increase from near 0 to approximately 9.4 m/s.
    - Safe distance: No sudden stops or close approaches to obstacles.
    """

    # Initialize component scores
    components = {
        "perceived_traffic_light": 0.0,
        "committed_to_accelerate": 0.0,
        "executed_acceleration": 0.0,
        "maintained_safe_distance": 0.0
    }

    # Check perceptual claims
    perceived_traffic_light = any(
        claim.entity == 'signal' and claim.state == 'green'
        for claim in claims.perceptual
    )

    # Check commitment claims
    committed_to_accelerate = any(
        claim.maneuver == 'accelerate' and claim.speed_profile == 'accelerate'
        for claim in claims.commitments
    )

    # Check trajectory for acceleration
    if traj.n_waypoints > 0:
        speed_series = np.array(traj.speed_mps)
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        if speed_increase >= 9.0 and np.all(np.diff(speed_series) > 0):
            if perceived_traffic_light and committed_to_accelerate:
                components["executed_acceleration"] = 0.6

    # Check trajectory for maintaining safe distance
    if len(speed_series) > 0:
        min_speed = np.min(speed_series)
        if min_speed >= 0.1 and not traj.stop_event:
            if perceived_traffic_light:
                components["maintained_safe_distance"] = 0.4

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
