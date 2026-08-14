"""clip 766f64e5-e272-447f-91c1-3c8023aa944e - attempt 3/5 - gate PASS (pos 0.70, max pert 0.14, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for evaluating the rollout:
    - lateral_avoidance: Credit for steering left to avoid the person directing traffic.
    """

    # Initialize component scores
    lateral_avoidance = 0.0

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor for lateral maneuver
        lateral_avoidance = 0.7 * min(1.0, lateral_offset_change / 1.5)  # Adjusted threshold to reflect significant lateral change

    return {
        "lateral_avoidance": lateral_avoidance
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
