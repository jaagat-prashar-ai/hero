"""clip fdef3d23-b2b1-4652-b402-0013da0d466e - attempt 1/5 - gate PASS (pos 0.70, max pert 0.14, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Merging right into the lane while maintaining a safe distance from traffic cones.
    - Perceptual mention of relevant entities.
    - Lateral maneuver execution with a graded trajectory factor.
    """

    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Check for perceptual mentions of relevant entities
    if any(p.entity in ('construction_cones', 'lane') for p in claims.perceptual):
        perceptual_score = 0.1

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Calculate the lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor for lateral maneuver execution
        lateral_maneuver_score = 0.6 * min(1.0, lateral_offset_change / 2.36)

    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver_execution": lateral_maneuver_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
