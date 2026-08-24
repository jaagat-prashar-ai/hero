"""clip 43d096ec-19ef-4388-8f6b-f0b476617b99 - attempt 5/5 - gate PASS (pos 0.79, max pert 0.10, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene 43d096ec-19ef-4388-8f6b-f0b476617b99:
    - Deceleration for traffic light and pedestrian crossing: speed drop >= 4.0 m/s, min speed timing
    - Perceptual mention of traffic light, pedestrian, or crosswalk
    """
    # Initialize component scores
    comp = {
        "decelerate_for_traffic_light": 0.0,
        "mention_traffic_light_or_pedestrian": 0.0
    }

    # Check for perceptual mentions
    if any(p.entity in ('signal', 'pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["mention_traffic_light_or_pedestrian"] = 0.1

    # Check for deceleration commitment and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop, floor at 4.0 m/s
        if speed_drop >= 4.0:
            # Check timing of minimum speed
            min_speed_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
            min_speed_time = min_speed_idx * traj.dt_s
            if 3.0 <= min_speed_time <= 4.0:  # Timing window for min speed
                comp["decelerate_for_traffic_light"] = 0.7 * min(1.0, speed_drop / 4.4)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
