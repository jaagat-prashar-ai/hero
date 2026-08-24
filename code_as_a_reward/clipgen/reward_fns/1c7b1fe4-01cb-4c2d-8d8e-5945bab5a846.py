"""clip 1c7b1fe4-01cb-4c2d-8d8e-5945bab5a846 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene 1c7b1fe4-01cb-4c2d-8d8e-5945bab5a846:
    - Decelerate for pedestrian crossing: speed drop >= 2.0 m/s by t=4.4s
    - Perceptual mention of pedestrian
    """
    # Initialize component scores
    comp = {
        "decelerate_for_pedestrian": 0.0,
        "mention_pedestrian": 0.0
    }

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Check for deceleration commitment and corresponding trajectory
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.0, 6.4))
        speed_drop = initial_speed - min_speed_after
        if speed_drop >= 2.0:
            comp["decelerate_for_pedestrian"] = 0.65 * min(1.0, speed_drop / 2.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
