"""clip 89083261-7dff-4809-9cb4-27671f04585d - attempt 2/5 - gate PASS (pos 0.90, max pert 0.10, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of steering left in response to temporary traffic delineators.
    Scene-derived thresholds:
    - Lateral offset: at least +0.33 m leftward
    - Heading change: at least -0.1 degrees leftward
    """

    # Initialize component scores
    perceptual_credit = 0.0
    lateral_maneuver_credit = 0.0

    # Check for perceptual mentions of traffic delineators
    if any(p.entity in ('construction_cones', 'barricades', 'work_zone') for p in claims.perceptual):
        perceptual_credit = 0.1  # Small additive weight for perceptual mention

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'turn', 'merge', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the graded lateral factor
        lateral_offset = traj.final_lateral_offset_m
        lateral_factor = 0.45 * min(1.0, max(0.0, (lateral_offset - 0.33) / 0.33))  # Graded factor for leftward offset
        heading_change = traj.total_heading_change_deg
        heading_factor = 0.35 * min(1.0, max(0.0, (-heading_change - 0.1) / 0.1))  # Graded factor for leftward heading change
        lateral_maneuver_credit = lateral_factor + heading_factor

    return {
        "perceptual_mention": perceptual_credit,
        "lateral_maneuver": lateral_maneuver_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
