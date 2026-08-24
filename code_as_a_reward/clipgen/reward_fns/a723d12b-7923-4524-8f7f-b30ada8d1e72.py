"""clip a723d12b-7923-4524-8f7f-b30ada8d1e72 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 8)"""
def components(claims, traj):
    """Calculate component contributions for the scene:
    1. Adapting speed for the upcoming curve.
    2. Maintaining safe distance from oncoming vehicle.
    Thresholds: speed drop >= 1.1 m/s, lateral offset >= -5.8 m.
    """
    # Initialize component scores
    comp = {
        "decelerate_execution": 0.0,
        "lateral_nudge_execution": 0.0,
    }

    # Commitment claims and trajectory checks
    # Deceleration for curve
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time = 2.2  # Time at which minimum speed occurs in the positive case
        if speed_drop >= 1.1 and np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s <= min_speed_time:
            comp["decelerate_execution"] = 0.7 * min(1.0, speed_drop / 2.2)

    # Lateral nudge for maintaining distance
    if any(c.maneuver in ('nudge', 'lane_change', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        lateral_offset = traj.final_lateral_offset_m
        if lateral_offset <= -5.8:
            comp["lateral_nudge_execution"] = 0.3 * min(1.0, abs(lateral_offset) / 11.64)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
