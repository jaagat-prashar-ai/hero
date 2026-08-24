"""clip b7601514-6757-4570-a676-6d694bca2035 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene:
    - Yield to pedestrians at the crosswalk: perceptual mention and deceleration.
    - Trajectory expectations: speed drop of at least 2.8 m/s, graded, with timing consideration.
    """
    components = {}

    # Perceptual mention of pedestrians or crosswalk
    saw_pedestrian = any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual)
    components['saw_pedestrian'] = 0.1 if saw_pedestrian else 0.0

    # Commitment to decelerate (stop/yield/wait/decelerate)
    committed_to_slow = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Trajectory speed drop with timing consideration
    speed_series = np.array(traj.speed_mps)
    initial_speed = traj.initial_speed_mps
    min_speed_after = np.min(window(speed_series, traj.dt_s, 3.0, traj.n_waypoints * traj.dt_s))
    speed_drop = initial_speed - min_speed_after

    # Graded factor for speed drop, requiring commitment to slow
    if committed_to_slow:
        # Ensure the minimum speed occurs towards the end of the window
        min_speed_time_idx = np.argmin(window(speed_series, traj.dt_s, 3.0, traj.n_waypoints * traj.dt_s))
        min_speed_time = 3.0 + min_speed_time_idx * traj.dt_s
        if min_speed_time >= 5.0:  # Ensure deceleration is sustained towards the end
            components['decelerate_executed'] = 0.6 * min(1.0, speed_drop / 5.6)
        else:
            components['decelerate_executed'] = 0.0
    else:
        components['decelerate_executed'] = 0.0

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
