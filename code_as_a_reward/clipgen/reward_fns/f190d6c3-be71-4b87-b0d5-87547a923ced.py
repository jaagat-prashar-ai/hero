"""clip f190d6c3-be71-4b87-b0d5-87547a923ced - attempt 2/5 - gate PASS (pos 0.70, max pert 0.12, real rollout argmax 3)"""
def components(claims, traj):
    """Components for the scene where the ego vehicle steers left through a construction zone.
    
    Decisive Events:
    1. Steering left through the construction zone, avoiding obstacles on the left.
    
    Scene-derived thresholds:
    - Lateral offset change of at least -2.1 m (half of the GT's -4.23 m).
    - Perceptual mention of construction-related entities.
    """
    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Check for perceptual mentions of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_score = 0.1  # Small additive weight for perceptual mention

    # Check for lateral maneuver commitment and matching trajectory execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor for lateral offset change, conditioned on commitment
        if lateral_offset_change > 0:  # Ensure leftward movement
            lateral_maneuver_score = 0.6 * min(1.0, abs(lateral_offset_change) / 4.23)

    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver_executed": lateral_maneuver_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
