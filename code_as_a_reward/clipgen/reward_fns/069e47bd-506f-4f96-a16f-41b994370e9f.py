"""clip 069e47bd-506f-4f96-a16f-41b994370e9f - attempt 1/5 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 10)"""
def components(claims, traj):
    """Components for scene 069e47bd-506f-4f96-a16f-41b994370e9f:
    - Maintain safe distance from a car occupying part of the lane by steering slightly to the right.
    - Perceptual mention of 'vehicle_generic' or 'lane'.
    - Lateral maneuver to the right with graded lateral offset change.
    - Trajectory thresholds: lateral offset change >= 0.16 m, heading change >= -1.65 degrees.
    """
    perceptual_vehicle = any(p.entity in ('vehicle_generic', 'lane') for p in claims.perceptual)
    commitment_lateral = any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments)

    # Trajectory analysis
    lateral_offset_change = abs(traj.final_lateral_offset_m)  # Use absolute to capture rightward movement
    heading_change = traj.total_heading_change_deg

    # Graded lateral factor
    lateral_factor = 0.5 * min(1.0, lateral_offset_change / 0.32) if commitment_lateral else 0.0
    heading_factor = 0.3 * min(1.0, abs(heading_change) / 3.3) if commitment_lateral else 0.0

    # Perceptual mention credit
    perceptual_credit = 0.2 if perceptual_vehicle else 0.0

    return {
        "perceptual_vehicle": perceptual_credit,
        "lateral_maneuver": lateral_factor,
        "heading_adjustment": heading_factor,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
