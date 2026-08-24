"""clip 96199a49-f1da-415b-bc79-8dcbc5d97726 - attempt 3/5 - gate PASS (pos 0.80, max pert 0.05, real rollout argmax 2)"""
def components(claims, traj):
    """Calculate component contributions for the scene where the expert steers left to avoid a construction zone with barriers. Key thresholds: lateral offset change >= 0.4 m, total heading change ~7 degrees."""
    
    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Perceptual component: mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_score = 0.05  # Reduced weight for mention-only credit

    # Lateral maneuver component: steering left
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate total heading change
        total_heading_change = traj.total_heading_change_deg
        # Graded factor for lateral movement
        if total_heading_change > 0:  # Ensure the maneuver is in the correct direction
            lateral_maneuver_score = 0.75 * min(1.0, total_heading_change / 7.0)

    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver": lateral_maneuver_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
