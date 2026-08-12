"""clip 2d99f4b2-c60c-4c66-ae99-2e0a8295abc1 - attempt 3/3 - gate PASS (pos 0.90, max pert 0.40, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event:
    Gentle deceleration to maintain a safe distance from the cyclist ahead.
    - Deceleration: Speed drop of approximately 1.0 to 1.5 m/s within the window.
    - Perceptual and commitment claims must align with trajectory execution.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check perceptual claims for detecting a cyclist ahead
    cyclist_detected = any(pc.entity == 'cyclist' and pc.state == 'ahead' for pc in claims.perceptual)
    if cyclist_detected:
        perceptual_score = 0.2

    # Check commitment claims for deceleration and maintaining safe distance
    deceleration_committed = any(cc.maneuver == 'decelerate' and cc.speed_profile == 'decelerate' for cc in claims.commitments)
    distance_maintained_committed = any(cc.maneuver == 'keep_distance' and cc.speed_profile == 'maintain' for cc in claims.commitments)
    if deceleration_committed:
        commitment_score += 0.2
    if distance_maintained_committed:
        commitment_score += 0.2

    # Check trajectory for gentle deceleration
    if traj.n_waypoints > 0:
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.final_speed_mps
        min_speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)  # Focus on the entire window
        min_speed = min(min_speed_window) if len(min_speed_window) > 0 else traj.initial_speed_mps

        # Check if the speed drop is within the expected range and occurs towards the end
        if 1.0 <= speed_drop <= 1.5 and min_speed == traj.final_speed_mps:
            if cyclist_detected and (deceleration_committed or distance_maintained_committed):
                trajectory_score = 0.5

    return {
        "perceptual_claims": perceptual_score,
        "commitment_claims": commitment_score,
        "trajectory_execution": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
