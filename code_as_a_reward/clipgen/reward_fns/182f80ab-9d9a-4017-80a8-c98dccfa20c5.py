"""clip 182f80ab-9d9a-4017-80a8-c98dccfa20c5 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.17, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring a rollout's faithfulness to the scene:
    - Steering left through a construction zone.
    - Perceptual mention of construction-related entities.
    - Lateral maneuver execution with leftward heading change and lateral offset.
    Scene-derived thresholds: min heading change +1.0 deg, min lateral offset +0.36 m.
    """
    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Check for relevant perceptual mentions
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_score = 0.1  # Small additive weight for perceptual mention

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate heading change and lateral offset
        heading_change = traj.total_heading_change_deg
        lateral_offset = traj.final_lateral_offset_m

        # Graded factor for heading change
        heading_change_factor = 0.45 * min(1.0, heading_change / 2.0)  # Half of GT's +2.0 degrees
        # Graded factor for lateral offset
        lateral_offset_factor = 0.45 * min(1.0, lateral_offset / 0.73)  # Half of GT's +0.73 meters

        # Combine graded factors for lateral maneuver score
        lateral_maneuver_score = heading_change_factor + lateral_offset_factor

    # Return component contributions
    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver_execution": lateral_maneuver_score
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
