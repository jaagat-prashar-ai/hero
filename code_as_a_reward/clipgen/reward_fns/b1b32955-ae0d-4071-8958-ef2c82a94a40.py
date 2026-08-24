"""clip b1b32955-ae0d-4071-8958-ef2c82a94a40 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene b1b32955-ae0d-4071-8958-ef2c82a94a40:
    - Decisive event: Steering left to pass a construction zone.
    - Perceptual mention: work_zone or related entities.
    - Commitment: Lateral maneuver (lane_change/nudge) to the left.
    - Trajectory: Leftward lateral offset change of at least 5.0 m.
    """
    perceptual_credit = 0.0
    lateral_commitment_credit = 0.0
    lateral_trajectory_credit = 0.0

    # Perceptual mention credit
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_credit = 0.1

    # Lateral maneuver commitment credit
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Trajectory lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        if lateral_offset_change < -5.0:
            lateral_commitment_credit = 0.3
            lateral_trajectory_credit = 0.4 * min(1.0, abs(lateral_offset_change) / 10.15)

    return {
        "perceptual_mention": perceptual_credit,
        "lateral_commitment": lateral_commitment_credit,
        "lateral_trajectory": lateral_trajectory_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
