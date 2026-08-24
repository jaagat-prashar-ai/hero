"""clip 5fb16ec3-53e6-4525-abc1-84f65e61990c - attempt 3/5 - gate PASS (pos 0.80, max pert 0.08, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with construction zone and lateral adjustment:
    - Maintain safe distance from construction zone with barriers on the right.
    - Slight leftward lateral adjustment to avoid construction zone.
    - Trajectory expectations: lateral offset change >= 0.085 m.
    """

    # Initialize component scores
    perceptual_construction_zone = 0.0
    lateral_adjustment = 0.0

    # Check for perceptual claims related to the construction zone
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades') for p in claims.perceptual):
        perceptual_construction_zone = 0.05  # Mention-only credit

    # Check for lateral commitment and corresponding trajectory adjustment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset change
        final_offset = traj.final_lateral_offset_m
        initial_offset = traj.lateral_offset_m[0] if traj.n_waypoints > 0 else 0.0
        offset_change = final_offset - initial_offset

        # Graded factor for lateral adjustment, gated on commitment
        if offset_change >= 0.085:
            lateral_adjustment = 0.75 * min(1.0, (offset_change - 0.085) / 1.675)  # Adjusted for the positive case's offset change

    return {
        "perceptual_construction_zone": perceptual_construction_zone,
        "lateral_adjustment": lateral_adjustment,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
