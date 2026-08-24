"""clip af77ae64-f29f-4a0a-bf4d-62baa9f6c4ec - attempt 1/5 - gate PASS (pos 0.95, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with stop sign and pedestrian yield:
    - Deceleration for stop sign: speed drop >= 4.0 m/s
    - Yielding to pedestrian: speed drop >= 4.0 m/s
    - Perceptual mentions: 'intersection', 'pedestrian'
    """
    comp = {
        "decelerate_for_stop_sign": 0.0,
        "decelerate_for_pedestrian": 0.0,
        "mention_intersection": 0.0,
        "mention_pedestrian": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity == 'intersection' for p in claims.perceptual):
        comp["mention_intersection"] = 0.05

    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Check for deceleration commitment
    deceleration_claimed = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Calculate speed drop
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps

    # Deceleration for stop sign
    if deceleration_claimed:
        comp["decelerate_for_stop_sign"] = 0.45 * min(1.0, speed_drop / 6.0)

    # Deceleration for pedestrian
    if deceleration_claimed:
        comp["decelerate_for_pedestrian"] = 0.45 * min(1.0, speed_drop / 6.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
