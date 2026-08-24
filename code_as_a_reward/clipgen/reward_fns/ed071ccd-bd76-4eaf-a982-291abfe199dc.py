"""clip ed071ccd-bd76-4eaf-a982-291abfe199dc - attempt 1/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene with deceleration due to pedestrian ahead.
    
    Decisive Events:
    1. Deceleration to maintain a safe distance from the pedestrian ahead.
       - Perceptual mention: 'pedestrian'
       - Commitment: speed_profile='decelerate'
       - Trajectory: speed drop of at least 0.45 m/s by t=5.8s
    2. Maintaining position with minimal lateral movement.
       - Trajectory: lateral offset within 0.015 m
    
    Scene-derived thresholds:
    - Speed drop: 0.45 m/s (half of 0.9 m/s)
    - Lateral offset: 0.015 m (half of 0.03 m)
    """
    perceptual_weight = 0.1
    deceleration_weight = 0.7
    lateral_weight = 0.2

    # Perceptual mention of pedestrian
    saw_pedestrian = perceptual_weight * any(
        p.entity in ('pedestrian',) for p in claims.perceptual
    )

    # Commitment to decelerate
    committed_to_decelerate = any(
        c.speed_profile == 'decelerate' for c in claims.commitments
    )

    # Trajectory speed drop
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    deceleration_executed = deceleration_weight * min(1.0, speed_drop / 0.9) if committed_to_decelerate else 0.0

    # Trajectory lateral offset
    max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
    maintaining_position = lateral_weight * (1.0 - min(1.0, max_lateral_offset / 0.03))

    return {
        "saw_pedestrian": saw_pedestrian,
        "deceleration_executed": deceleration_executed,
        "maintaining_position": maintaining_position,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
