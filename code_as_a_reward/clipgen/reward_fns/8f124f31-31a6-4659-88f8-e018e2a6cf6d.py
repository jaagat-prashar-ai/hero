"""clip 8f124f31-31a6-4659-88f8-e018e2a6cf6d - attempt 2/5 - gate PASS (pos 1.00, max pert 0.37, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene 8f124f31-31a6-4659-88f8-e018e2a6cf6d:
    - Steering left to avoid a construction zone with lane barriers.
      - Perceptual mention: {'work_zone', 'construction_cones', 'barricades'}
      - Commitment: 'lane_change' or 'nudge' (left)
      - Trajectory: Lateral offset change of at least +1.29 m
    - Maintaining speed:
      - Perceptual mention: {'road', 'traffic', 'clear_path'}
      - Commitment: 'accelerate' (optional)
      - Trajectory: Speed increase of at least +2.2 m/s
    """
    perceptual_weight = 0.1
    commitment_weight = 0.45
    trajectory_weight = 0.45

    # Perceptual mention credit
    perceptual_mention = 0.0
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades') for p in claims.perceptual):
        perceptual_mention += perceptual_weight

    # Commitment and trajectory for steering left
    lateral_commitment = any(
        c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right'
        for c in claims.commitments
    )
    lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
    lateral_trajectory = 0.0
    if lateral_commitment:
        lateral_trajectory = commitment_weight * min(1.0, lateral_offset_change / 2.58)

    # Commitment and trajectory for maintaining speed
    speed_commitment = any(c.speed_profile == 'accelerate' for c in claims.commitments)
    speed_increase = traj.final_speed_mps - traj.initial_speed_mps
    speed_trajectory = 0.0
    if speed_commitment:
        speed_trajectory = trajectory_weight * min(1.0, speed_increase / 4.4)

    return {
        "perceptual_mention": perceptual_mention,
        "lateral_trajectory": lateral_trajectory,
        "speed_trajectory": speed_trajectory,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
