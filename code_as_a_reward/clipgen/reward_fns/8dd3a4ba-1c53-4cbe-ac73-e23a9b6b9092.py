"""clip 8dd3a4ba-1c53-4cbe-ac73-e23a9b6b9092 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scene with a decisive event of steering left to pass a parked vehicle.
    - Commitment family: lateral maneuvers (lane_change, nudge, merge) excluding right
    - Trajectory expectations: leftward lateral offset change of at least 0.8 m
    """

    # Initialize component scores
    lateral_maneuver = 0.0

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge') and c.direction != 'right' for c in claims.commitments):
        # Calculate the lateral offset change
        lateral_offsets = np.array(traj.lateral_offset_m)
        initial_offset = lateral_offsets[0]
        max_left_offset = np.min(lateral_offsets)
        lateral_offset_change = initial_offset - max_left_offset

        # Graded factor for lateral maneuver
        lateral_maneuver = 0.7 * min(1.0, lateral_offset_change / 1.62)

    return {
        "lateral_maneuver": lateral_maneuver,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
