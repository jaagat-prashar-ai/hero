"""clip b30fbc40-aa45-4553-b121-229ecf8c3947 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 10)"""
def components(claims, traj):
    """Components for scene b30fbc40-aa45-4553-b121-229ecf8c3947:
    - Steering left to pass a parked vehicle while maintaining a safe distance from right-side vehicles.
    - Trajectory expectations: leftward offset of at least -0.4 m, heading change of at least -2.3 degrees.
    - Perceptual mentions: vehicle_generic, work_zone, barricades.
    """
    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('vehicle_generic', 'work_zone', 'barricades') for p in claims.perceptual):
        perceptual_score = 0.1

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset and heading change
        lateral_offset = traj.final_lateral_offset_m
        heading_change = traj.total_heading_change_deg

        # Graded lateral offset factor
        lateral_offset_factor = 0.5 * min(1.0, abs(lateral_offset) / 0.85)

        # Graded heading change factor
        heading_change_factor = 0.4 * min(1.0, abs(heading_change) / 4.6)

        # Combine factors for lateral maneuver score
        lateral_maneuver_score = lateral_offset_factor + heading_change_factor

    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver_executed": lateral_maneuver_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
