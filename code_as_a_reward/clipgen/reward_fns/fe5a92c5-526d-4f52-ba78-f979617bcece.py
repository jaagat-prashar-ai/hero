"""clip fe5a92c5-526d-4f52-ba78-f979617bcece - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for reward calculation based on the scene's decisive events:
    - Steering right to maintain a safe distance from the construction zone.
    - Trajectory should show a rightward heading change of at least -1.8 degrees.
    - Perceptual mention of construction-related entities.
    """
    # Initialize component scores
    perceptual_mention = 0.0
    lateral_maneuver = 0.0

    # Check for perceptual mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_mention = 0.1  # Small weight for mention

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Calculate the rightward heading change
        heading_change = traj.total_heading_change_deg
        # Ensure the heading change is rightward (negative) and graded
        if heading_change < 0:
            lateral_maneuver = 0.6 * min(1.0, abs(heading_change) / 3.6)

    return {
        "perceptual_mention": perceptual_mention,
        "lateral_maneuver": lateral_maneuver,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
