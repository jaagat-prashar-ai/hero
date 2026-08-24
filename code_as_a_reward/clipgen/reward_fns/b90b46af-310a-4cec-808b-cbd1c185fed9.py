"""clip b90b46af-310a-4cec-808b-cbd1c185fed9 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.15, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene b90b46af-310a-4cec-808b-cbd1c185fed9:
    - Decisive event: Steering right to follow temporary traffic delineators.
    - Perceptual mention: Any of 'construction_cones', 'barricades', 'work_zone'.
    - Commitment: Lateral maneuver (nudge/lane_change/merge/turn) with rightward direction.
    - Trajectory: Rightward lateral offset change and heading change, graded above 0.4 m and 4.5 degrees.
    """
    perceptual_credit = 0.1 if any(p.entity in ('construction_cones', 'barricades', 'work_zone')
                                   for p in claims.perceptual) else 0.0

    lateral_commitment = any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and
                             c.direction != 'left' for c in claims.commitments)

    # Calculate the rightward lateral offset change
    lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
    graded_lateral_factor = 0.3 * min(1.0, max(0.0, lateral_offset_change / 0.8))

    # Calculate the heading change
    heading_change_factor = 0.4 * min(1.0, max(0.0, traj.total_heading_change_deg / 9.0))

    lateral_execution_credit = (graded_lateral_factor + heading_change_factor) if lateral_commitment else 0.0

    return {
        "perceptual_mention": perceptual_credit,
        "lateral_execution": lateral_execution_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
