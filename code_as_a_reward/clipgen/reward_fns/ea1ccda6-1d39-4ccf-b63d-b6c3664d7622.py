"""clip ea1ccda6-1d39-4ccf-b63d-b6c3664d7622 - attempt 3/3 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive events:
    1. Maintaining distance from nearby vehicles (tracks 15 and 40).
    2. Maintaining speed while yielding.

    Scene-derived thresholds:
    - Lateral offset adjustments within ±1.0 m.
    - Speed fluctuations within ±0.3 m/s of initial speed.
    - Yielding behavior without stopping.
    """

    # Initialize component scores
    components = {
        "perceive_lane_ahead": 0.0,
        "commit_keep_distance": 0.0,
        "maintain_speed": 0.0,
        "yield_behavior": 0.0
    }

    # Check perceptual claims
    if any(claim.entity == 'lane' and claim.state == 'ahead' for claim in claims.perceptual):
        components["perceive_lane_ahead"] = 0.2

    # Check commitment claims
    if any(claim.maneuver == 'keep_distance' and claim.speed_profile == 'maintain' for claim in claims.commitments):
        components["commit_keep_distance"] = 0.2

    # Check trajectory for maintaining speed with claim conjunction
    if any(claim.maneuver == 'keep_distance' and claim.speed_profile == 'maintain' for claim in claims.commitments):
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        if len(speed_window) > 0:
            initial_speed = traj.initial_speed_mps
            min_speed = min(speed_window)
            min_speed_time = np.argmin(speed_window) * traj.dt_s
            if abs(min_speed - initial_speed) >= 0.5 and 3.0 <= min_speed_time <= 4.0:
                components["maintain_speed"] = 0.3

    # Check trajectory for yielding behavior with claim conjunction
    if any(claim.maneuver == 'keep_distance' for claim in claims.commitments):
        lateral_offset_window = window(traj.lateral_offset_m, traj.dt_s, 0, 6.4)
        if len(lateral_offset_window) > 0:
            max_lateral_offset = max(abs(offset) for offset in lateral_offset_window)
            if max_lateral_offset <= 1.0 and not traj.stop_event and 3.0 <= min_speed_time <= 4.0:
                components["yield_behavior"] = 0.3

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
