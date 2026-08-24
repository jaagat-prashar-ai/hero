"""clip f276423c-a333-4ce6-b33a-3742fe7930f0 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene with pedestrians crossing and trailers on the left.
    - Deceleration to maintain safe distance from pedestrians.
    - Trajectory expectations: speed drop of at least 0.45 m/s within the first 4 seconds.
    - Perceptual mentions: 'pedestrian'.
    """
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrians": 0.0,
    }

    # Perceptual mention component
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Trajectory-based commitment component
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Deceleration for pedestrians with timing consideration
        if speed_drop >= 0.45 and min_speed_time >= 3.0:
            comp["decelerate_for_pedestrians"] = 0.6 * min(1.0, speed_drop / 0.9)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
