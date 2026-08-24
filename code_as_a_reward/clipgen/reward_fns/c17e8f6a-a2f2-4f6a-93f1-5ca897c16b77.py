"""clip c17e8f6a-a2f2-4f6a-93f1-5ca897c16b77 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.19, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene c17e8f6a-a2f2-4f6a-93f1-5ca897c16b77:
    - Yield to pedestrian (speed_profile='decelerate', mention 'pedestrian')
    - Automobiles on the right (mention 'vehicle_generic')
    """
    comp = {
        "mention_pedestrian": 0.0,
        "yield_to_pedestrian": 0.0,
        "mention_vehicle_generic": 0.0,
    }

    # Perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Commitment to yield (decelerate) to pedestrian
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = speed_series.min()
        speed_drop = initial_speed - min_speed_after
        # Graded factor for speed drop, focusing on timing
        if min_speed_after < initial_speed:
            comp["yield_to_pedestrian"] = 0.7 * min(1.0, speed_drop / 3.8)

    # Perceptual mention of vehicles
    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        comp["mention_vehicle_generic"] = 0.1

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
