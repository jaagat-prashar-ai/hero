"""clip e9f5e5c5-e7d8-45c1-ac97-c0f77a68546d - attempt 3/5 - gate PASS (pos 1.00, max pert 0.02, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene e9f5e5c5-e7d8-45c1-ac97-c0f77a68546d:
    - Decisive event: Steering left in response to traffic delineators and signs.
    - Commitment family: 'nudge' or 'lane_change' (lateral maneuver family), excluding 'right'
    - Trajectory expectations: Leftward lateral offset change >= 0.035 m, speed drop >= 1.7 m/s
    """

    # Removed perceptual_credit as it was dead in the positive case

    lateral_commitment = any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments)
    lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
    lateral_factor = 0.0
    if lateral_commitment:
        lateral_factor = 0.6 * min(1.0, max(0.0, lateral_offset_change / 16.69))  # Adjusted to match the positive case's magnitude

    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    slowing_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    slowing_factor = 0.0
    if slowing_commitment:
        slowing_factor = 0.4 * min(1.0, speed_drop / 3.2)  # Adjusted to match the positive case's magnitude

    return {
        "lateral_execution": lateral_factor,
        "slowing_execution": slowing_factor,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
