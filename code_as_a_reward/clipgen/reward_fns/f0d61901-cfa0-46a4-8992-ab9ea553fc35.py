"""clip f0d61901-cfa0-46a4-8992-ab9ea553fc35 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.16, real rollout argmax 0)"""
def components(claims, traj):
    """Components for reward calculation based on decisive events:
    - Steering left due to temporary traffic delineators.
    - Scene-derived thresholds: lateral offset change >= 0.5 m to the left.
    """
    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Check for perceptual mention of related entities
    if any(p.entity in ('construction_cones', 'barricades', 'work_zone', 'lane') for p in claims.perceptual):
        perceptual_score = 0.1  # Small additive weight for mention

    # Check for lateral maneuver commitment to steer left
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor for lateral movement to the left
        lateral_maneuver_score = 0.6 * min(1.0, max(0.0, lateral_offset_change / 0.5))

    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver_executed": lateral_maneuver_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
