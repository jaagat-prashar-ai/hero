"""clip 0c7917da-3bd6-4e24-b7c5-8525ae9d0f4a - attempt 4/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with a lateral shift to avoid a partially blocking car.
    - Perceptual: Mention of 'vehicle_generic' as the obstacle.
    - Commitment: Lateral maneuver (lane_change/nudge) to the left.
    - Trajectory: Lateral offset change of at least +0.29 m by the end of the window.
    """
    perceptual_weight = 0.1
    lateral_weight = 0.9

    # Perceptual component: mention of vehicle_generic
    saw_vehicle = any(p.entity == 'vehicle_generic' for p in claims.perceptual)
    perceptual_score = perceptual_weight if saw_vehicle else 0.0

    # Commitment component: lateral maneuver to the left
    lateral_commitment = any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments)

    # Trajectory component: graded lateral offset change
    final_offset = traj.final_lateral_offset_m
    lateral_offset_change = max(0.0, -final_offset)  # Assuming initial offset is 0.0
    lateral_score = lateral_weight * min(1.0, lateral_offset_change / 0.29) if lateral_commitment else 0.0

    return {
        "saw_vehicle": perceptual_score,
        "lateral_maneuver": lateral_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
