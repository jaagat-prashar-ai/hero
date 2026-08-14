"""clip 1b83cb72-8913-4641-8741-1943a345df8f - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event:
    - Navigating the construction zone by steering slightly left.
    - Perceptual mention of construction-related entities.
    - Lateral maneuver execution with a leftward offset.
    - Scene-derived thresholds: lateral offset floor at ~0.36 m.
    """

    # Initialize component scores
    perceptual_credit = 0.0
    lateral_maneuver_credit = 0.0

    # Check for perceptual mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_credit = 0.05  # Reduced weight for mention-only

    # Check for lateral maneuver commitment and execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the lateral offset factor
        lateral_offsets = window(traj.lateral_offset_m, traj.dt_s, 0.0, 6.4)
        max_lateral_offset = np.max(np.abs(lateral_offsets))
        lateral_factor = 0.65 * min(1.0, max_lateral_offset / 0.72)  # Graded factor based on max offset

        # Ensure the trajectory reflects a leftward maneuver
        if traj.total_heading_change_deg < 0:
            lateral_maneuver_credit = lateral_factor

    # Return component scores
    return {
        "perceptual_mention": perceptual_credit,
        "lateral_maneuver_execution": lateral_maneuver_credit
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
