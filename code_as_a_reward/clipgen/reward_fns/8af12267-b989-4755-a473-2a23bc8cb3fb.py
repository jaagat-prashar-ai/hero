"""clip 8af12267-b989-4755-a473-2a23bc8cb3fb - attempt 1/5 - gate PASS (pos 0.90, max pert 0.18, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with steering left to follow traffic barrels.
    
    Decisive Event: Steering left to follow traffic barrels.
    - Perceptual: Mention of 'barricades', 'construction_cones', or 'work_zone'.
    - Commitment: Lateral maneuver ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') excluding 'right'.
    - Trajectory: Lateral offset increase of at least 1.7 m, heading change of at least 3.7 degrees.
    """
    perceptual_credit = 0.1 if any(p.entity in ('barricades', 'construction_cones', 'work_zone') for p in claims.perceptual) else 0.0

    lateral_commitment = any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments)
    
    lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
    lateral_factor = 0.5 * min(1.0, lateral_offset_change / 3.4) if lateral_commitment else 0.0

    heading_change = traj.total_heading_change_deg
    heading_factor = 0.3 * min(1.0, heading_change / 7.4) if lateral_commitment else 0.0

    return {
        "perceptual_mention": perceptual_credit,
        "lateral_execution": lateral_factor,
        "heading_execution": heading_factor,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
