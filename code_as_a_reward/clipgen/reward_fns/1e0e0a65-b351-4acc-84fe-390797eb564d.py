"""clip 1e0e0a65-b351-4acc-84fe-390797eb564d - attempt 2/5 - gate PASS (pos 0.70, max pert 0.20, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    - Merge maneuver to the right indicated by traffic cones and road sign.
    - Trajectory should show a rightward heading change and lateral offset.
    - Perceptual mention of construction-related entities conditioned on a commitment.
    """

    # Initialize component scores
    perceptual_mention = 0.0
    lateral_maneuver = 0.0

    # Check for lateral maneuver commitment
    has_lateral_commitment = any(
        c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left'
        for c in claims.commitments
    )

    # Check for perceptual mention of construction-related entities conditioned on a commitment
    if has_lateral_commitment and any(
        p.entity in ('construction_cones', 'barricades', 'work_zone') for p in claims.perceptual
    ):
        perceptual_mention = 0.2

    # Calculate the graded trajectory factor for heading change
    heading_change = traj.total_heading_change_deg
    if has_lateral_commitment:
        lateral_maneuver = 0.5 * min(1.0, max(0.0, (heading_change - 0.5) / 0.5))

    return {
        "perceptual_mention": perceptual_mention,
        "lateral_maneuver": lateral_maneuver,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
