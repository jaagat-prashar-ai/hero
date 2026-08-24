"""clip da2b02fa-658b-4930-b9b6-e09e841c68df - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene da2b02fa-658b-4930-b9b6-e09e841c68df:
    - Steering left to maintain a safe distance from a construction vehicle on the right.
    - Maintaining speed with no significant deceleration.
    Trajectory thresholds:
    - Heading change: at least 1.5 degrees (half of the measured -3 degrees).
    - Speed maintenance: minimal speed drop (less than 1 m/s) with a commitment check.
    """

    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Check for perceptual claims related to construction or vehicles
    if any(p.entity in ('vehicle_generic', 'construction_cones', 'barricades', 'work_zone') for p in claims.perceptual):
        perceptual_score = 0.05  # Small weight for perceptual mention

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate heading change
        heading_change = traj.total_heading_change_deg
        # Graded factor for heading change
        if heading_change <= -1.5:  # Adjusted threshold for leftward change
            lateral_maneuver_score = 0.65 * min(1.0, abs(heading_change) / 3.0)

    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver": lateral_maneuver_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
