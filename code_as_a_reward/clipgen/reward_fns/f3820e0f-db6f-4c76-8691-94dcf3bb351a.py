"""clip f3820e0f-db6f-4c76-8691-94dcf3bb351a - attempt 1/5 - gate PASS (pos 0.70, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene:
    1. Stop Sign and Pedestrian: Deceleration commitment and speed drop.
    2. Protruding Object and Nearby Vehicles: Lateral nudge and offset change.
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    perceptual_stop_sign = 0.0
    deceleration_commitment = 0.0
    lateral_nudge_commitment = 0.0

    # Check for perceptual claims
    if any(p.entity in ('pedestrian', 'intersection') for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    if any(p.entity == 'intersection' for p in claims.perceptual):
        perceptual_stop_sign = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop
        deceleration_commitment = 0.5 * min(1.0, speed_drop / 0.4)

    # Check for lateral nudge commitment
    if any(c.maneuver in ('nudge', 'lane_change', 'merge') and c.direction != 'left' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = abs(traj.final_lateral_offset_m)
        # Graded factor for lateral offset change
        lateral_nudge_commitment = 0.3 * min(1.0, lateral_offset_change / 5.65)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "perceptual_stop_sign": perceptual_stop_sign,
        "deceleration_commitment": deceleration_commitment,
        "lateral_nudge_commitment": lateral_nudge_commitment,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
