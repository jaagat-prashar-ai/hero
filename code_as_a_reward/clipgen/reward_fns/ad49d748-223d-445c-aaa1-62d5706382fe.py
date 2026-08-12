"""clip ad49d748-223d-445c-aaa1-62d5706382fe - attempt 2/3 - gate PASS (pos 1.00, max pert 0.20, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of merging back into the lane.
    Scene-derived thresholds:
    - Lateral offset change: final offset within ±0.5 m of -3.52 m
    - Maximum lateral offset: within ±0.5 m of 3.59 m
    - Timing: Rightward steer should occur within the 6.4-second window
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check perceptual claims
    saw_lane = any(pc.entity == 'lane' for pc in claims.perceptual)
    saw_work_zone = any(pc.entity == 'work_zone' for pc in claims.perceptual)

    if saw_lane and saw_work_zone:
        perceptual_score = 0.1

    # Check commitment claims
    committed_to_merge = any(cc.maneuver == 'merge' for cc in claims.commitments)

    if committed_to_merge:
        commitment_score = 0.1

    # Check trajectory execution with conjunction
    if traj.n_waypoints > 0 and committed_to_merge:
        # Check lateral offset change
        final_offset = traj.final_lateral_offset_m
        max_offset = max(abs(offset) for offset in traj.lateral_offset_m)

        # Ensure the trajectory shows a rightward turn
        total_turn = traj.total_heading_change_deg

        if abs(final_offset + 3.52) <= 0.5 and total_turn < 0:
            trajectory_score += 0.4

        if abs(max_offset - 3.59) <= 0.5 and total_turn < 0:
            trajectory_score += 0.4

    # Total score is the sum of all components
    return {
        "perceptual_claims": perceptual_score,
        "commitment_claims": commitment_score,
        "trajectory_execution": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
