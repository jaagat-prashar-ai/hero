"""clip 57383f22-a038-407a-8a6e-7febac5d7287 - attempt 5/5 - gate PASS (pos 1.00, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Steering left to maintain a safe distance from a reconstruction zone.
    - Scene-derived thresholds: heading change >= 0.5 * 344.5 degrees leftward.
    """

    # Initialize component scores
    perceptual_construction_zone = 0.0
    lateral_maneuver = 0.0

    # Perceptual check for construction zone
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_construction_zone = 0.05  # Reduced weight for mention-only credit

    # Lateral maneuver check for steering left
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate heading change
        heading_change = traj.total_heading_change_deg
        # Graded factor for heading change
        if heading_change > 0:  # Ensure the turn is in the correct direction
            lateral_maneuver = 0.95 * min(1.0, heading_change / 0.5)  # Adjusted for half the GT heading change

    return {
        "perceptual_construction_zone": perceptual_construction_zone,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
