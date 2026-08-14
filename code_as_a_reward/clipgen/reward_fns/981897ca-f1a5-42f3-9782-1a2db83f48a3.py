"""clip 981897ca-f1a5-42f3-9782-1a2db83f48a3 - attempt 1/5 - gate PASS (pos 0.78, max pert 0.28, real rollout argmax 4)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive events:
    1. Steering left to return to the lane after passing vehicles.
       - Perceptual: vehicle_generic
       - Commitment: lane_change/nudge (left)
       - Trajectory: leftward heading change >= 2.4 degrees
    2. Maintaining safe distance from cyclist.
       - Perceptual: cyclist
       - No specific commitment required
       - Trajectory: speed increase >= 1.3 m/s
    """

    # Initialize component scores
    perceptual_vehicle = 0.0
    perceptual_cyclist = 0.0
    lateral_maneuver = 0.0
    speed_maintenance = 0.0

    # Check perceptual claims
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        perceptual_vehicle = 0.05  # Small weight for mentioning vehicles

    if any(p.entity == 'cyclist' for p in claims.perceptual):
        perceptual_cyclist = 0.05  # Small weight for mentioning cyclist

    # Check commitment claims and corresponding trajectory
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Graded lateral factor based on heading change
        heading_change = traj.total_heading_change_deg
        lateral_maneuver = 0.5 * min(1.0, abs(heading_change) / 4.8)  # GT heading change is -4.8 degrees

    # Check speed maintenance
    speed_increase = traj.final_speed_mps - traj.initial_speed_mps
    speed_maintenance = 0.4 * min(1.0, speed_increase / 2.6)  # GT speed increase is 2.6 m/s

    return {
        "perceptual_vehicle": perceptual_vehicle,
        "perceptual_cyclist": perceptual_cyclist,
        "lateral_maneuver": lateral_maneuver,
        "speed_maintenance": speed_maintenance
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
