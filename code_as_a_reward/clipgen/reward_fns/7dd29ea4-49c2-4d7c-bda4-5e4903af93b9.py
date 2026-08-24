"""clip 7dd29ea4-49c2-4d7c-bda4-5e4903af93b9 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene 7dd29ea4-49c2-4d7c-bda4-5e4903af93b9:
    - Decisive event: Steering left following temporary traffic delineators.
    - Trajectory expectations: Leftward heading change of at least -29.0 degrees.
    - Commitment family: Lateral maneuver ('turn', 'nudge') to the left.
    - Perceptual mention: Road features related to lane guidance or road boundaries.
    """
    # Initialize component scores
    perceptual_mention = 0.0
    lateral_maneuver = 0.0

    # Check for perceptual mentions of road features
    if any(p.entity in ('construction_cones', 'barricades', 'work_zone') for p in claims.perceptual):
        perceptual_mention = 0.05

    # Check for lateral maneuver commitment to the left
    if any(c.maneuver in ('turn', 'nudge', 'lane_change', 'merge', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the leftward heading change
        heading_change = traj.total_heading_change_deg
        # Graded trajectory factor for heading change
        lateral_maneuver = 0.65 * min(1.0, max(0.0, (-heading_change - 29.0) / 29.0))

    return {
        "perceptual_mention": perceptual_mention,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
