"""clip d47f16e0-198b-4593-9bc3-6285fb5371fa - attempt 2/5 - gate PASS (pos 1.00, max pert 0.30, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Commitment to a lateral maneuver (leftward) with graded heading change.
    Scene-derived thresholds: heading change >= +3.5 degrees.
    """

    # Commitment to a lateral maneuver (leftward) with trajectory execution
    lateral_commitment = any(
        c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right'
        for c in claims.commitments
    )
    heading_change = traj.total_heading_change_deg
    graded_heading_change = 0.7 * min(1.0, heading_change / 7.0) if lateral_commitment else 0.0

    # Combine components
    return {
        "lateral_commitment": 0.3 if lateral_commitment else 0.0,
        "graded_heading_change": graded_heading_change
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
