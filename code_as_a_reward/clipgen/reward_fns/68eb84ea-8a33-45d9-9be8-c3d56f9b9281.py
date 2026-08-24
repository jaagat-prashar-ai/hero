"""clip 68eb84ea-8a33-45d9-9be8-c3d56f9b9281 - attempt 4/5 - gate PASS (pos 0.95, max pert 0.50, real rollout argmax 1)"""
def components(claims, traj):
    """Components for navigating through a construction zone while maintaining a straight path.
    
    Decisive Event: Navigating the Construction Zone
    - Perceptual: Mention of construction-related entities.
    - Commitment: Maintain a straight path through the zone (lateral maneuver family).
    - Trajectory: Maintain or slightly increase speed, slight lateral offset, and minor heading adjustment.
    """
    # Initialize component scores
    perceptual_score = 0.0
    lateral_commitment_score = 0.0
    heading_adjustment_score = 0.0

    # Perceptual check: Mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_score = 0.05  # Reduced weight for mention-only credit

    # Commitment check: Lateral maneuver family (nudge, lane_change, etc.)
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Trajectory check: Slight lateral offset
        lateral_offset_change = abs(traj.final_lateral_offset_m - traj.lateral_offset_m[0])
        lateral_commitment_score = 0.45 * min(1.0, lateral_offset_change / 0.70)

    # Commitment check: Heading adjustment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') for c in claims.commitments):
        # Trajectory check: Minor heading adjustment
        if traj.total_heading_change_deg > 0:
            heading_adjustment_score = 0.50 * min(1.0, abs(traj.total_heading_change_deg) / 2.0)

    # Return component scores as a dictionary
    return {
        "perceptual_mention": perceptual_score,
        "lateral_commitment": lateral_commitment_score,
        "heading_adjustment": heading_adjustment_score
    }

def reward(claims, traj):
    # Calculate the total score and clamp it between 0.0 and 1.0
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
