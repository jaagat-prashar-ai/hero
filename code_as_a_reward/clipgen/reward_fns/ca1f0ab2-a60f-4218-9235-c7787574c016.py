"""clip ca1f0ab2-a60f-4218-9235-c7787574c016 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    """Components for reward function based on decisive events:
    1. Navigating through the construction zone: Expect mention of 'work_zone' or 'barricades' and a lateral maneuver like 'nudge' or 'lane_change'. Trajectory should maintain a lateral offset of at least -0.5 m.
    """
    # Initialize component scores
    lateral_maneuver = 0.0

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset factor
        final_offset = traj.final_lateral_offset_m
        total_turn = traj.total_heading_change_deg
        if total_turn > 0:  # Ensure the turn direction matches the expected positive turn
            lateral_factor = max(0.0, min(1.0, (final_offset + 0.1) / 0.23))  # Graded factor based on offset
            lateral_maneuver = 0.7 * lateral_factor

    return {
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
