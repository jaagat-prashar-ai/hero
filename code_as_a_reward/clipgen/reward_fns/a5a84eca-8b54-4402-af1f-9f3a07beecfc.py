"""clip a5a84eca-8b54-4402-af1f-9f3a07beecfc - attempt 1/5 - gate PASS (pos 0.70, max pert 0.15, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Merging to the left lane after passing a construction zone.
    - Perceptual mentions of construction-related entities.
    - Lateral maneuver execution with graded lateral offset change.
    - Trajectory thresholds are set at half the GT's magnitudes for flexibility.
    """

    # Initialize component scores
    perceptual_construction = 0.0
    lateral_maneuver = 0.0

    # Check for perceptual mentions related to the construction zone
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_construction = 0.1

    # Check for lateral maneuver commitment and corresponding trajectory execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor based on the lateral offset change
        lateral_maneuver = 0.6 * min(1.0, lateral_offset_change / 4.0)

    return {
        "perceptual_construction": perceptual_construction,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
