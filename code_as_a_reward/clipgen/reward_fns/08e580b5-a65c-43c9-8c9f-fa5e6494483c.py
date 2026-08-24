"""clip 08e580b5-a65c-43c9-8c9f-fa5e6494483c - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 08e580b5-a65c-43c9-8c9f-fa5e6494483c:
    - Decelerate for pedestrian at crosswalk: speed drop >= 2.7 m/s
    - Perceptual mention of pedestrian or crosswalk
    """
    comp = {
        "decelerate_for_pedestrian": 0.0,
        "mention_pedestrian_or_crosswalk": 0.0,
    }

    # Perceptual mention of pedestrian or crosswalk
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["mention_pedestrian_or_crosswalk"] = 0.1

    # Decelerate for pedestrian at crosswalk
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, 6.4))
        speed_drop = initial_speed - min_speed_after
        comp["decelerate_for_pedestrian"] = 0.6 * min(1.0, speed_drop / 5.4)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
