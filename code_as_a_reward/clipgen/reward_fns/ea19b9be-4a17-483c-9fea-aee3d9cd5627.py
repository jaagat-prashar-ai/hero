"""clip ea19b9be-4a17-483c-9fea-aee3d9cd5627 - attempt 4/5 - gate PASS (pos 0.90, max pert 0.40, real rollout argmax 2)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene:
    - Temporary Lane Navigation: Expect mention of construction-related entities and a lateral maneuver commitment.
    - Trajectory should show minimal lateral movement consistent with navigating a temporary lane.
    - Scene-derived thresholds: Lateral offset change floor at 0.5 m, graded factor for lateral movement.
    """
    perceptual_credit = 0.0
    lateral_commitment_credit = 0.0
    lateral_trajectory_credit = 0.0

    # Perceptual check for construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_credit = 0.1  # Small additive weight for mentioning construction-related entities

    # Commitment check for lateral maneuvering
    lateral_commitment = any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments)
    if lateral_commitment:
        lateral_commitment_credit = 0.3  # Larger weight for correct lateral maneuver commitment

        # Trajectory check for lateral movement, gated by commitment
        lateral_offsets = window(traj.lateral_offset_m, traj.dt_s, 0.0, 6.4)
        if len(lateral_offsets) > 0:
            lateral_offset_change = abs(lateral_offsets[-1] - lateral_offsets[0])
            # Ensure significant lateral movement and correct timing
            if lateral_offset_change > 0.5 and np.argmin(window(traj.speed_mps, traj.dt_s, 0.0, 6.4)) * traj.dt_s < 3.2:
                lateral_trajectory_credit = 0.5 * min(1.0, lateral_offset_change / 1.0)  # Graded factor for lateral movement

    return {
        "perceptual_credit": perceptual_credit,
        "lateral_commitment_credit": lateral_commitment_credit,
        "lateral_trajectory_credit": lateral_trajectory_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
