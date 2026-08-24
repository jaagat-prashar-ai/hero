"""clip a3261aed-4554-4f0e-8a14-4eefcd75159b - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 10)"""
def components(claims, traj):
    """Components for scene a3261aed-4554-4f0e-8a14-4eefcd75159b:
    - Steering right following temporary traffic delineators
    Trajectory thresholds: rightward heading change >= -12.0 degrees.
    """

    # Initialize component scores
    lateral_maneuver_right = 0.0

    # Check for lateral maneuver commitments to the right
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Calculate rightward heading change
        heading_change = traj.total_heading_change_deg
        if heading_change <= -12.0:
            lateral_maneuver_right = 0.7 * min(1.0, abs(heading_change) / 24.0)

    return {
        "lateral_maneuver_right": lateral_maneuver_right
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
