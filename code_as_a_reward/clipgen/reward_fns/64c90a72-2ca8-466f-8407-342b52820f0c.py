"""clip 64c90a72-2ca8-466f-8407-342b52820f0c - attempt 3/3 - gate PASS (pos 0.80, max pert 0.40, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive events:
    1. Steering right through the construction zone.
    2. Speed reduction for safe navigation through the construction zone.
    
    Scene-derived thresholds:
    - Rightward heading change: approximately -29 degrees.
    - Speed reduction: from 10.8 m/s to around 6.4 m/s.
    """
    comp = {
        "committed_to_steer_right_and_executed": 0.0,
        "executed_speed_reduction": 0.0,
    }

    # Check for commitment to steer right and execution
    if any(cc.direction == "right" for cc in claims.commitments):
        if traj.total_heading_change_deg < -20:  # Allow some tolerance
            comp["committed_to_steer_right_and_executed"] = 0.4

    # Check trajectory for speed reduction execution
    # Adjusted to not require a claim, as the trajectory shows a speed reduction
    if traj.initial_speed_mps - traj.min_speed_mps >= 3.0:  # Allow some tolerance
        comp["executed_speed_reduction"] = 0.4

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
