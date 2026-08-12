"""clip c8e6054a-d331-4964-9fe8-45c96be0cf0d - attempt 2/3 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of yielding to a pedestrian.
    The scene-derived thresholds are based on the expert's behavior:
    - Speed reduction of at least 7.0 m/s, reaching a minimum speed of 0.5 m/s or lower.
    - Speed reduction should occur between t=5.0 s and t=6.0 s.
    - Lateral offset should remain within |0.5| m.
    """

    # Initialize component scores
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check for commitment claims
    yield_commitment = any(cc.maneuver == 'yield' and cc.speed_profile == 'decelerate' for cc in claims.commitments)

    if yield_commitment:
        commitment_score = 0.3

    # Check trajectory for speed reduction with timing
    if traj.n_waypoints > 0:
        speed_window = window(traj.speed_mps, traj.dt_s, 5.0, 6.0)
        min_speed_in_window = speed_window.min() if len(speed_window) > 0 else float('inf')
        speed_drop = traj.initial_speed_mps - min_speed_in_window

        if speed_drop >= 7.0 and min_speed_in_window <= 0.5:
            trajectory_score += 0.2

        # Check lateral offset
        lateral_offset_window = window(traj.lateral_offset_m, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)
        max_lateral_offset = max(abs(lateral_offset_window.min()), abs(lateral_offset_window.max()))
        
        if max_lateral_offset <= 0.5:
            trajectory_score += 0.2

    # Conjunction: Require both a commitment claim and matching trajectory execution
    if yield_commitment and speed_drop >= 7.0 and min_speed_in_window <= 0.5:
        trajectory_score += 0.3

    return {
        "commitment_claims": commitment_score,
        "trajectory_execution": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
