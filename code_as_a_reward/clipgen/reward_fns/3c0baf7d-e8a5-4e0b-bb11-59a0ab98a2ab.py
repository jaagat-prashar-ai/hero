"""clip 3c0baf7d-e8a5-4e0b-bb11-59a0ab98a2ab - attempt 1/5 - gate PASS (pos 0.89, max pert 0.40, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with stopping behind a lead vehicle and maintaining safe distance from a motorcycle.
    
    Decisive Events:
    1. Stop behind the lead vehicle (Track 72) with minimal speed increase.
    2. Maintain safe lateral distance from the motorcycle (Track 62) with minimal lateral offset.
    
    Scene-derived thresholds:
    - Speed should remain close to 0.0 m/s (minimal increase).
    - Lateral offset should remain close to 0.0 m (minimal deviation).
    """
    # Initialize component scores
    scores = {
        "perceptual_vehicle": 0.0,
        "perceptual_motorcycle": 0.0,
        "stop_executed": 0.0,
        "maintain_lateral_position": 0.0
    }

    # Perceptual mentions
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        scores["perceptual_vehicle"] = 0.05

    if any(p.entity == 'cyclist' for p in claims.perceptual):
        scores["perceptual_motorcycle"] = 0.05

    # Commitment to stop or decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Graded speed factor: minimal speed increase
        max_speed = max(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        scores["stop_executed"] = 0.5 * min(1.0, (0.2 - max_speed) / 0.2)

    # Lateral position maintenance
    max_lateral_offset = max(abs(window(traj.lateral_offset_m, traj.dt_s, 0, 6.4)))
    scores["maintain_lateral_position"] = 0.4 * min(1.0, (0.05 - max_lateral_offset) / 0.05)

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
