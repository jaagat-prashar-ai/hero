"""clip f0947d1e-482a-44c9-94f1-3bdff177dc8c - attempt 2/3 - gate PASS (pos 0.90, max pert 0.50, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of yielding to a pedestrian at a crosswalk.
    Thresholds are inspired by the expert's behavior in the scene:
    - Speed reduction of at least 2.5 m/s within the first 5.5 seconds.
    - Lateral offset adjustment within ±2.5 m.
    - Commitment to yield with a deceleration profile.
    """
    # Initialize component scores
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check commitment claims
    committed_to_yield = any(cc.maneuver == 'yield' and cc.speed_profile == 'decelerate' for cc in claims.commitments)
    if committed_to_yield:
        commitment_score = 0.3

    # Check trajectory execution
    if traj.n_waypoints > 0:
        # Speed reduction check with timing
        initial_speed = traj.initial_speed_mps
        min_speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4).min() if len(window(traj.speed_mps, traj.dt_s, 0, 6.4)) > 0 else initial_speed
        speed_reduction = initial_speed - min_speed_window
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s

        if speed_reduction >= 2.5 and min_speed_time <= 6.4 and committed_to_yield:
            trajectory_score += 0.4

        # Lateral offset check
        final_lateral_offset = traj.final_lateral_offset_m
        if abs(final_lateral_offset) <= 2.5:
            trajectory_score += 0.2

    return {
        "commitment_score": commitment_score,
        "trajectory_score": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
