"""clip bb891322-799e-416f-8987-3d3919d3c817 - attempt 1/5 - gate PASS (pos 0.86, max pert 0.40, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive events:
    1. Traffic Light Turning Green: Expect mention of 'signal' and a speed increase.
    2. Maintaining Lane Position: Expect mention of 'lane' and minimal lateral deviation.
    Trajectory thresholds are derived from the ground truth dossier, with generous floors.
    """

    # Initialize component scores
    comp = {
        "mention_signal": 0.0,
        "accelerate_executed": 0.0,
        "mention_lane": 0.0,
        "maintain_lane_executed": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity == 'signal' for p in claims.perceptual):
        comp["mention_signal"] = 0.1

    if any(p.entity == 'lane' for p in claims.perceptual):
        comp["mention_lane"] = 0.1

    # Check for acceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Calculate speed increase
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        # Graded factor for acceleration execution
        comp["accelerate_executed"] = 0.5 * min(1.0, speed_increase / 1.0)

    # Check for maintaining lane position through trajectory
    max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
    # Graded factor for maintaining lane execution
    comp["maintain_lane_executed"] = 0.3 * min(1.0, (3.0 - max_lateral_offset) / 3.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
