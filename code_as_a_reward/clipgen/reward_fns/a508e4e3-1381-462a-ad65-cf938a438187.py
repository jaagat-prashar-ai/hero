"""clip a508e4e3-1381-462a-ad65-cf938a438187 - attempt 1/5 - gate PASS (pos 0.95, max pert 0.55, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene a508e4e3-1381-462a-ad65-cf938a438187:
    - Yield to traffic on the main road: expect deceleration with a speed drop of at least 0.4 m/s.
    - Keep a safe distance from the pedestrian: expect lateral offset maintenance of at least +0.5 m.
    - Perceptual mentions: 'vehicle_generic' for traffic, 'pedestrian' for pedestrian.
    """
    comp = {
        "yield_to_traffic": 0.0,
        "safe_distance_pedestrian": 0.0,
        "mention_traffic": 0.0,
        "mention_pedestrian": 0.0,
    }

    # Perceptual mentions
    if any(p.entity in ('vehicle_generic', 'cross_traffic') for p in claims.perceptual):
        comp["mention_traffic"] = 0.05

    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Yield to traffic on the main road
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 0.4:
            comp["yield_to_traffic"] = 0.5 * min(1.0, speed_drop / 0.8)

    # Keep a safe distance from the pedestrian
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        lateral_offset = abs(traj.final_lateral_offset_m)
        if lateral_offset >= 0.5:
            comp["safe_distance_pedestrian"] = 0.4 * min(1.0, lateral_offset / 0.99)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
