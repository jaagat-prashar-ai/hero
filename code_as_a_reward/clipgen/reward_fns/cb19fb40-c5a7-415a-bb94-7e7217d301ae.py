"""clip cb19fb40-c5a7-415a-bb94-7e7217d301ae - attempt 4/5 - gate PASS (pos 1.00, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the decisive event of steering left to pass a pedestrian.
    - Commitment: Lateral maneuver to the left (nudge/lane_change) to avoid the pedestrian.
    - Trajectory: Graded lateral offset change.
    """

    # Initialize component scores
    lateral_commitment_score = 0.0
    lateral_offset_score = 0.0

    # Commitment component: Lateral maneuver to the left
    if any(c.maneuver in ('lane_change', 'nudge') and c.direction != 'right' for c in claims.commitments):
        # Graded lateral offset change
        lateral_offset_change = abs(traj.final_lateral_offset_m)
        if lateral_offset_change >= 0.15:  # Ensure a significant leftward movement
            lateral_offset_score = 0.6 * min(1.0, lateral_offset_change / 0.30)
            lateral_commitment_score = 0.4

    return {
        "lateral_commitment": lateral_commitment_score,
        "lateral_offset": lateral_offset_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
