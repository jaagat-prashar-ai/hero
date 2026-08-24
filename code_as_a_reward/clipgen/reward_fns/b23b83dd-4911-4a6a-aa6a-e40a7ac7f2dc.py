"""clip b23b83dd-4911-4a6a-aa6a-e40a7ac7f2dc - attempt 3/5 - gate PASS (pos 0.99, max pert 0.05, real rollout argmax 10)"""
def components(claims, traj):
    """Components for scene b23b83dd-4911-4a6a-aa6a-e40a7ac7f2dc:
    - Steering left following temporary traffic delineators
    Thresholds:
    - Heading change: at least -20 degrees
    """

    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Check for perceptual mentions related to lane or vehicles
    if any(p.entity in ('lane', 'vehicle_generic') for p in claims.perceptual):
        perceptual_score = 0.05  # Small additive weight for mention

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'turn', 'merge', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate heading change
        heading_change = traj.total_heading_change_deg
        # Graded factor for heading change
        if heading_change < -20:  # Ensure leftward change
            lateral_maneuver_score = 0.95 * min(1.0, abs(heading_change) / 40.0)

    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver_executed": lateral_maneuver_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
