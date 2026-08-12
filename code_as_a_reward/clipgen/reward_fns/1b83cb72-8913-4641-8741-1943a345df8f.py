"""clip 1b83cb72-8913-4641-8741-1943a345df8f - attempt 2/3 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 4)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of navigating
    through the construction zone on the right by steering slightly to the left.
    Scene-derived thresholds:
    - Lateral offset: +0.5 m to +1.0 m left
    - Speed: 0.7 m/s to 10.0 m/s final
    - Heading change: +0.5 to +1.5 degrees
    """
    comp = {
        "steer_left_conjunction": 0.0,
        "speed_trajectory_shape": 0.0
    }

    # Check for conjunction of steering left claim and trajectory execution
    steer_left_claim = any(cc.direction == 'left' for cc in claims.commitments)
    final_offset = traj.final_lateral_offset_m
    if steer_left_claim and 0.5 <= final_offset <= 1.0:
        comp["steer_left_conjunction"] = 0.5

    # Check for speed trajectory shape (increase over time)
    if traj.n_waypoints > 0:
        speed_series = np.array(traj.speed_mps)
        if np.all(np.diff(speed_series) > -0.1):  # Allow slight noise
            comp["speed_trajectory_shape"] = 0.5

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
