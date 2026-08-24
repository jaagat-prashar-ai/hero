"""clip 614440d5-515b-452d-86ce-fff5af2bc688 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.12, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with construction zone and right-side obstacles.
    
    Decisive events:
    1. Steer left to maintain a safe distance from the construction zone with traffic cones on the right.
       - Perceptual mention: {'work_zone', 'construction_cones', 'barricades'}
       - Commitment: 'nudge' or 'lane_change' (to the left)
       - Trajectory: Lateral offset change of at least +1.45 m (half of the positive case's +2.89 m)
    """
    # Initialize component scores
    perceptual_construction = 0.0
    lateral_execution = 0.0

    # Perceptual mentions
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades') for p in claims.perceptual):
        perceptual_construction = 0.1

    # Lateral commitment and execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge') and c.direction != 'right' for c in claims.commitments):
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        lateral_execution = 0.6 * min(1.0, lateral_offset_change / 2.89)

    return {
        "perceptual_construction": perceptual_construction,
        "lateral_execution": lateral_execution
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
