"""clip 3ccff50d-289b-46ab-ae15-ab5570f8beca - attempt 3/3 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 2)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "perceived_construction_zone": 0.0,
        "committed_to_steer_right": 0.0,
        "steer_right_conjunction": 0.0,
        "executed_speed_adjustment": 0.0
    }

    # Check for perception of construction zone
    if any(pc.entity == 'construction_cones' for pc in claims.perceptual):
        scores["perceived_construction_zone"] = 0.1

    # Check for commitment to steer right
    if any(cc.direction == 'right' for cc in claims.commitments):
        scores["committed_to_steer_right"] = 0.1

    # Conjunction: Check for both commitment and execution of steering right
    if scores["committed_to_steer_right"] > 0 and -4.0 < traj.final_lateral_offset_m < -3.0:
        scores["steer_right_conjunction"] = 0.5

    # Check for execution of speed adjustment
    speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
    if len(speed_window) > 0:
        min_speed = min(speed_window)
        if 6.5 <= min_speed <= 6.7:
            scores["executed_speed_adjustment"] = 0.3

    return scores

def reward(claims, traj):
    """Reward function for scene with decisive events: steering right through construction zone and speed adjustment."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
