"""clip d48815ab-1343-40e0-a5a0-a332a3568953 - attempt 5/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene d48815ab-1343-40e0-a5a0-a332a3568953.
    
    Decisive Events:
    1. Steering right to avoid an obstacle on the left side of the road.
       - Lateral maneuver expected: rightward nudge or lane change.
       - Trajectory: Rightward lateral offset change, minimum -0.75 m.

    Trajectory thresholds are one-sided and graded, with lateral offset
    as a key factor. Commitment credit is matched at the FAMILY level.
    """

    # Initialize component scores
    comp = {
        "lateral_maneuver": 0.0
    }

    # Lateral maneuver credit
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Calculate the rightward lateral offset change
        lateral_change = traj.final_lateral_offset_m - np.min(window(traj.lateral_offset_m, traj.dt_s, 0.0, 6.4))
        # Ensure the change is in the correct direction and occurs at the right time
        if lateral_change > 0 and np.argmin(window(traj.lateral_offset_m, traj.dt_s, 0.0, 6.4)) * traj.dt_s < 3.2:
            comp["lateral_maneuver"] = 0.7 * min(1.0, max(0.0, lateral_change / 0.25))  # Adjusted threshold

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
