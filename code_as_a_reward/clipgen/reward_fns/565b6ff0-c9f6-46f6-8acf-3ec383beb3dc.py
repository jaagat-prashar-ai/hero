"""clip 565b6ff0-c9f6-46f6-8acf-3ec383beb3dc - attempt 3/5 - gate PASS (pos 0.96, max pert 0.27, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene 565b6ff0-c9f6-46f6-8acf-3ec383beb3dc:
    1. Steering left to maintain a safe distance from the construction zone.
       - Perceptual mention: 'work_zone', 'construction_cones', 'barricades', 'workers'
       - Commitment: Lateral maneuver ('nudge', 'lane_change') excluding 'right'
       - Trajectory: Leftward lateral offset change, min 0.5 m
    """
    perceptual_weight = 0.1
    lateral_weight = 0.9

    # Perceptual mention for construction zone
    saw_construction = perceptual_weight * any(
        p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers')
        for p in claims.perceptual
    )

    # Lateral maneuver commitment
    lateral_commitment = any(
        c.maneuver in ('lane_change', 'nudge') and c.direction != 'right'
        for c in claims.commitments
    )

    # Trajectory: Leftward lateral offset change
    lateral_offset_change = traj.final_lateral_offset_m - min(traj.lateral_offset_m)
    lateral_factor = lateral_weight * min(1.0, max(0.0, lateral_offset_change / 0.43)) if lateral_commitment else 0.0

    return {
        "saw_construction": saw_construction,
        "lateral_factor": lateral_factor,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
