"""clip 924c8ee5-b0f9-4ffb-a836-aa3a8203c022 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.07, real rollout argmax 8)"""
def components(claims, traj):
    """
    Components for scene 924c8ee5-b0f9-4ffb-a836-aa3a8203c022:
    - Steering adjustment to avoid the white van (lateral maneuver to the left).
    Trajectory thresholds:
    - Lateral offset change: at least +1.14 m (half of +2.28 m).
    """

    # Initialize component scores
    perceptual_credit = 0.0
    lateral_maneuver_credit = 0.0

    # Check for perceptual mention of vehicle_generic
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        perceptual_credit = 0.05

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        lateral_maneuver_credit = 0.65 * min(1.0, lateral_offset_change / 2.28)

    return {
        "perceptual_mention": perceptual_credit,
        "lateral_maneuver_executed": lateral_maneuver_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
