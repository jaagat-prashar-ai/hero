"""clip 98006e5e-4d02-4c37-ac10-fa4a6d24ffc4 - attempt 2/5 - gate PASS (pos 0.79, max pert 0.11, real rollout argmax 2)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene:
    - Lane change to the right in response to a construction zone.
    - Perceptual recognition of construction-related entities.
    - Trajectory showing a rightward lateral offset and heading change.
    """

    # Initialize component scores
    perceptual_construction = 0.0
    lane_change_right = 0.0

    # Perceptual check for construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_construction = 0.1

    # Commitment check for lane change to the right
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Trajectory check for rightward lane change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        heading_change = traj.total_heading_change_deg

        # Graded trajectory factor for lateral offset change
        if lateral_offset_change > 0:
            lateral_factor = 0.5 * min(1.0, lateral_offset_change / 1.5)
        else:
            lateral_factor = 0.0

        # Graded trajectory factor for heading change
        if heading_change > 0:
            heading_factor = 0.2 * min(1.0, heading_change / 0.5)
        else:
            heading_factor = 0.0

        # Combine trajectory factors
        lane_change_right = lateral_factor + heading_factor

    return {
        "perceptual_construction": perceptual_construction,
        "lane_change_right": lane_change_right,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
