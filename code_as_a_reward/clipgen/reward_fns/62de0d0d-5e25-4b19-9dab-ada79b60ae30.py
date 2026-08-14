"""clip 62de0d0d-5e25-4b19-9dab-ada79b60ae30 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 62de0d0d-5e25-4b19-9dab-ada79b60ae30:
    - Maintain speed while following the lead vehicle and navigating through the construction zone.
    - Trajectory expectations: speed drop >= 0.1 m/s, lateral offset <= 0.5 m.
    """
    perceptual_weight = 0.1
    commitment_weight = 0.6
    lateral_weight = 0.3

    # Perceptual claim: mention of lead vehicle or construction zone
    perceptual_mention = any(p.entity in ('lead_vehicle', 'work_zone', 'construction_cones')
                             for p in claims.perceptual)
    perceptual_score = perceptual_weight if perceptual_mention else 0.0

    # Commitment claim: maintain speed
    maintain_speed_commitment = any(c.speed_profile == 'maintain' for c in claims.commitments)

    # Trajectory: speed maintenance
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    speed_maintenance_score = 0.0
    if maintain_speed_commitment:
        speed_maintenance_score = commitment_weight * min(1.0, speed_drop / 0.2)

    # Trajectory: lateral offset within bounds
    max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
    lateral_offset_score = 0.0
    if max_lateral_offset <= 0.5:
        lateral_offset_score = lateral_weight * (0.5 / 0.5)  # Full credit if within bounds

    return {
        "perceptual_mention": perceptual_score,
        "speed_maintenance": speed_maintenance_score,
        "lateral_offset": lateral_offset_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
