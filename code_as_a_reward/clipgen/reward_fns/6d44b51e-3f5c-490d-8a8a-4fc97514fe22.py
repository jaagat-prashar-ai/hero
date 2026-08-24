"""clip 6d44b51e-3f5c-490d-8a8a-4fc97514fe22 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with steering adjustment to follow traffic delineators.
    
    Decisive event: Steering right to follow temporary traffic delineators.
    - Perceptual: Mention of road infrastructure entities like 'construction_cones', 'barricades', 'work_zone'.
    - Commitment: Lateral maneuver from the family (nudge, lane_change, merge, turn, enter, exit) with direction not 'left'.
    - Trajectory: Rightward lateral offset change of at least 0.6 m, with graded credit for larger changes.
    """
    perceptual_credit = 0.05 * any(p.entity in ('construction_cones', 'barricades', 'work_zone') for p in claims.perceptual)
    
    lateral_commitment = any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments)
    
    # Calculate lateral offset change
    lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
    lateral_factor = 0.65 * min(1.0, lateral_offset_change / 1.2) if lateral_commitment and lateral_offset_change > 0.6 else 0.0
    
    return {
        "perceptual_mention": perceptual_credit,
        "lateral_execution": lateral_factor
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
