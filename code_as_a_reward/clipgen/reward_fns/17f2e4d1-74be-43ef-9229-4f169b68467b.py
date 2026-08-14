"""clip 17f2e4d1-74be-43ef-9229-4f169b68467b - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 8)"""
def components(claims, traj):
    """Components for scoring a rollout based on steering through a construction zone and maintaining speed.
    
    Decisive events:
    1. Steering right through the construction zone.
       - Perceptual mention: construction-related entities.
       - Commitment: lateral maneuver (lane_change/nudge/turn) excluding left.
       - Trajectory: slight rightward adjustment (heading change).
    
    Scene-derived thresholds:
    - Heading change: <= -0.2 degrees (graded).
    """

    # Initialize component scores
    perceptual_construction = 0.0
    lateral_maneuver = 0.0

    # Check perceptual claims for construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_construction = 0.05  # Reduced weight for mention-only credit

    # Check for lateral maneuver commitment and corresponding trajectory
    if any(c.maneuver in ('lane_change', 'nudge', 'turn') and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change <= -0.2:
            lateral_maneuver = 0.65 * min(1.0, abs(heading_change) / 1.0)  # Increased weight and adjusted threshold

    return {
        "perceptual_construction": perceptual_construction,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
