"""clip 814c97d9-2ea4-40b1-b55b-f6a8f967e8cf - attempt 1/5 - gate PASS (pos 0.90, max pert 0.12, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 814c97d9-2ea4-40b1-b55b-f6a8f967e8cf:
    Navigate through a construction zone with barriers, maintaining a steady trajectory.
    - Perceptual: Mention of construction-related entities.
    - Commitment: Lateral maneuver (nudge/lane_change) excluding right.
    - Trajectory: Lateral offset change and heading change reflecting navigation.
    """
    perceptual_credit = 0.1 if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual) else 0.0

    lateral_commitment = any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments)
    lateral_offset_change = abs(traj.final_lateral_offset_m) - abs(traj.lateral_offset_m[0])
    lateral_trajectory_credit = 0.5 * min(1.0, lateral_offset_change / 0.9) if lateral_commitment else 0.0

    heading_change = traj.total_heading_change_deg
    heading_trajectory_credit = 0.3 * min(1.0, abs(heading_change) / 3.8) if lateral_commitment else 0.0

    return {
        "perceptual_mention": perceptual_credit,
        "lateral_maneuver_executed": lateral_trajectory_credit,
        "heading_adjustment_executed": heading_trajectory_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
