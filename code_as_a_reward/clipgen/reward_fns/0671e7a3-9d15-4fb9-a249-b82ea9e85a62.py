"""clip 0671e7a3-9d15-4fb9-a249-b82ea9e85a62 - attempt 3/5 - gate PASS (pos 0.90, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 0671e7a3-9d15-4fb9-a249-b82ea9e85a62:
    - Slight rightward nudge to avoid construction area on the left.
    - Trajectory thresholds: lateral offset change ~0.25 m rightward.
    """

    # Initialize component scores
    perceptual_construction = 0.0
    lateral_nudge_right = 0.0

    # Perceptual claims
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_construction = 0.1

    # Commitment claims and trajectory execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Calculate the rightward lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor for rightward nudge, conditioned on both claim and trajectory
        if lateral_offset_change > 0.125:  # Ensure a significant rightward nudge
            lateral_nudge_right = 0.8 * min(1.0, lateral_offset_change / 0.25)

    return {
        "perceptual_construction": perceptual_construction,
        "lateral_nudge_right": lateral_nudge_right,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
