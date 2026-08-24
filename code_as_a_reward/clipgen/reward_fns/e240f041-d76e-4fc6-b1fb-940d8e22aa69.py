"""clip e240f041-d76e-4fc6-b1fb-940d8e22aa69 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.06, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene e240f041-d76e-4fc6-b1fb-940d8e22aa69:
    1. Steering left to maintain a safe distance from the construction zone.
       - Perceptual mention of construction-related entities.
       - Lateral maneuver to the left (nudge/lane_change/turn) excluding right.
       - Lateral offset change of at least 0.025 m.
    """
    # Initialize component scores
    comp = {
        "perceptual_construction": 0.0,
        "lateral_maneuver": 0.0
    }

    # Perceptual mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.05

    # Lateral maneuver to the left
    if any(c.maneuver in ('lane_change', 'nudge', 'turn') and c.direction != 'right' for c in claims.commitments):
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        if lateral_offset_change > 0.025:
            comp["lateral_maneuver"] = 0.65 * min(1.0, lateral_offset_change / 2.94)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
