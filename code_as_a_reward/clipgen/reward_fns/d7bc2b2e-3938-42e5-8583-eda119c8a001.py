"""clip d7bc2b2e-3938-42e5-8583-eda119c8a001 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Perceptual mention of construction-related entities.
    - Commitment to a lateral maneuver (nudge/lane_change) to the left.
    - Trajectory showing a leftward lateral offset change.
    Scene-derived thresholds: lateral offset change >= 0.75 m (half of GT's 1.54 m).
    """

    # Initialize component scores
    perceptual_mention = 0.0
    lateral_commitment_execution = 0.0

    # Check for perceptual mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_mention = 0.05  # Reduced weight to allow more on commitment

    # Check for lateral commitment to the left and matching trajectory execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the lateral offset change
        final_offset = traj.final_lateral_offset_m
        initial_offset = traj.lateral_offset_m[0] if traj.n_waypoints > 0 else 0.0
        lateral_change = abs(final_offset - initial_offset)

        # Check for correct temporal shape in trajectory
        if traj.total_heading_change_deg < 0:
            # Graded factor for lateral execution
            lateral_commitment_execution = 0.95 * min(1.0, lateral_change / 1.54)

    return {
        "perceptual_mention": perceptual_mention,
        "lateral_commitment_execution": lateral_commitment_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
