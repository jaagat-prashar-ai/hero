"""clip 107e7a76-169a-4d03-92c4-d7b667c312e0 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene 107e7a76-169a-4d03-92c4-d7b667c312e0:
    - Steering right through construction zone: lateral maneuver with minimal rightward heading change.
    - Speed maintenance: significant deceleration.
    """

    # Initialize component scores
    comp = {
        "lateral_maneuver": 0.0,
        "speed_maintenance": 0.0,
    }

    # Lateral maneuver: rightward steering through construction zone
    if any(c.maneuver in ('lane_change', 'nudge', 'turn', 'merge', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change < 0:  # Rightward change
            comp["lateral_maneuver"] = 0.3 * min(1.0, abs(heading_change) / 1.5)

    # Speed maintenance: significant deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        comp["speed_maintenance"] = 0.7 * min(1.0, speed_drop / 2.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
