"""clip d6bd9a24-aa65-44c4-b31d-38e0bda599ae - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene d6bd9a24-aa65-44c4-b31d-38e0bda599ae:
    - Steering right to maintain a safe distance from the construction zone.
    Trajectory thresholds:
    - Heading change: at least 1.5 degrees (half of 3 degrees).
    """

    # Initialize component scores
    perceptual_construction = 0.0
    lateral_maneuver = 0.0

    # Check for perceptual claims
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_construction = 0.05  # Mention-only credit

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change > 1.5:  # Ensure significant rightward change
            lateral_maneuver = 0.65 * min(1.0, heading_change / 3.0)

    return {
        "perceptual_construction": perceptual_construction,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
