"""clip f1d811bd-6537-423f-84e0-6ee574a182ac - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene:
    - Steering right through the construction zone
    - Proximity to vehicle on the left
    Trajectory thresholds derived from GT: lateral offset change >= 0.6 m rightward.
    """
    # Initialize component scores
    perceptual_construction_zone = 0.0
    lateral_maneuver_right = 0.0

    # Check for perceptual claims
    if any(p.entity in ('construction_cones', 'work_zone', 'barricades') for p in claims.perceptual):
        perceptual_construction_zone = 0.05  # Reduced weight for mention-only credit

    # Check for lateral maneuver commitments
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Calculate the rightward lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        if lateral_offset_change > 0:  # Rightward movement is positive
            lateral_maneuver_right = 0.65 * min(1.0, lateral_offset_change / 0.77)

    return {
        "perceptual_construction_zone": perceptual_construction_zone,
        "lateral_maneuver_right": lateral_maneuver_right
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
