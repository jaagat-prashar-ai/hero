"""clip e44655a1-ba92-4187-bf72-d491fe2c738e - attempt 4/5 - gate PASS (pos 0.74, max pert 0.16, real rollout argmax 10)"""
def components(claims, traj):
    """Components for scene with lane change to avoid construction zone:
    - Lane change to the right: lateral offset change, commitment to lane_change
    - Avoidance of construction zone: perceptual mention of construction-related entities
    Thresholds derived from GT: lateral offset change >= 1.0 m, direction 'right' only.
    """

    # Initialize component scores
    comp = {
        "perceptual_construction_zone": 0.0,
        "lane_change_executed": 0.0,
    }

    # Perceptual mention of construction zone
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction_zone"] = 0.1

    # Commitment to lane change to the right
    lane_change_commitment = any(
        c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left'
        for c in claims.commitments
    )

    # Trajectory analysis for lane change
    lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
    if lane_change_commitment and lateral_offset_change < 0:
        comp["lane_change_executed"] = 0.7 * min(1.0, abs(lateral_offset_change) / 2.12)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
