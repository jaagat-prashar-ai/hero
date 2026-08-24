"""clip 7bad5757-6abe-4c0b-bb3c-de4690ae0ce3 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.13, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene 7bad5757-6abe-4c0b-bb3c-de4690ae0ce3:
    - Steering left through a construction zone, requiring lateral maneuver.
    - Perceptual mentions of construction-related entities.
    Trajectory thresholds:
    - Lateral offset increase to at least +4.95 m.
    """

    # Initialize component scores
    perceptual_construction = 0.0
    lateral_maneuver = 0.0

    # Check for perceptual mentions of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_construction = 0.1

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset increase
        lateral_offset_increase = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        lateral_maneuver = 0.9 * min(1.0, lateral_offset_increase / 9.89)

    return {
        "perceptual_construction": perceptual_construction,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
