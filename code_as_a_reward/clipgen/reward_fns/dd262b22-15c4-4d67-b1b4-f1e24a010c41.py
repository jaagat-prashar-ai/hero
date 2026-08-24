"""clip dd262b22-15c4-4d67-b1b4-f1e24a010c41 - attempt 1/5 - gate PASS (pos 0.80, max pert 0.03, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene dd262b22-15c4-4d67-b1b4-f1e24a010c41:
    - Steering right to follow temporary traffic barriers.
    - Perceptual mention of barriers or related entities.
    - Lateral maneuver with rightward direction.
    - Graded trajectory factors for heading change and lateral offset.
    """
    # Initialize component scores
    perceptual_mention = 0.0
    lateral_maneuver = 0.0
    heading_change_factor = 0.0
    lateral_offset_factor = 0.0

    # Perceptual mention of barriers or related entities
    if any(p.entity in ('barricades', 'construction_cones', 'work_zone') for p in claims.perceptual):
        perceptual_mention = 0.1

    # Lateral maneuver with rightward direction
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Graded trajectory factor for heading change
        heading_change = traj.total_heading_change_deg
        if heading_change < 0:  # Rightward turn
            heading_change_factor = 0.5 * min(1.0, abs(heading_change) / 4.0)  # GT heading change is -4.1 deg

        # Graded trajectory factor for lateral offset
        final_offset = traj.final_lateral_offset_m
        if final_offset < 0:  # Rightward offset
            lateral_offset_factor = 0.3 * min(1.0, abs(final_offset) / 3.96)  # GT final offset is -3.96 m

        lateral_maneuver = heading_change_factor + lateral_offset_factor

    return {
        "perceptual_mention": perceptual_mention,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
