"""clip 087130e1-01d0-4263-9dbd-e589fe785e9e - attempt 3/5 - gate PASS (pos 1.00, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene with pedestrian yielding and oncoming vehicle deceleration.
    
    Decisive Events:
    1. Yield to the pedestrian crossing the crosswalk.
    2. Decelerate to maintain a safe distance from the oncoming vehicle.
    
    Trajectory Expectations:
    - Speed drop of at least 3.0 m/s (half of GT drop of 6.3 m/s).
    - Minimum speed occurring around t=3.8 s.
    """

    # Initialize component scores
    components = {
        "perceptual_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
        "decelerate_for_vehicle": 0.0,
    }

    # Check perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        components["perceptual_pedestrian"] = 0.05

    # Check commitment claims and trajectory for pedestrian
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
        if speed_drop >= 3.0 and 3.0 <= min_speed_time <= 4.5:
            components["decelerate_for_pedestrian"] = 0.55 * min(1.0, speed_drop / 6.0)

    # Check commitment claims and trajectory for oncoming vehicle
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
        if speed_drop >= 3.0 and 3.0 <= min_speed_time <= 4.5:
            components["decelerate_for_vehicle"] = 0.40 * min(1.0, speed_drop / 6.0)

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
