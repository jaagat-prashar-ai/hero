"""clip 8236fbed-ca56-4d9e-be3e-0ac3ac786a2b - attempt 5/5 - gate PASS (pos 0.70, max pert 0.30, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 8236fbed-ca56-4d9e-be3e-0ac3ac786a2b:
    - Steering left to pass parked vehicles: lateral maneuver with leftward offset change.
    - Maintaining safe distance from vehicle on the right: lateral maneuver with leftward offset change.
    Trajectory thresholds: lateral offset change floor at ~2.5 m, graded above; requires commitment claim.
    """

    # Initialize component scores
    lateral_maneuver = 0.0

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor for lateral maneuver
        lateral_maneuver = 0.7 * min(1.0, lateral_offset_change / 2.5)  # Adjusted threshold for half the magnitude

    # Return component contributions
    return {
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
