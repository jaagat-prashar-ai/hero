"""clip 4a90f0d4-cef6-424e-ab13-d266f8fa79d8 - attempt 3/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scene with crossing pedestrians at a crosswalk.
    
    Decisive Event: Pedestrians at the crosswalk prompting deceleration.
    - Perceptual mention: 'pedestrian' or 'crosswalk'
    - Commitment: speed_profile='decelerate'
    - Trajectory: Speed drop of at least 0.2 m/s within the window, with timing consideration.
    """
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_commitment_execution": 0.0
    }

    # Perceptual mention credit
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Combined commitment and trajectory check for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        if traj.n_waypoints > 0:
            speed_drop = traj.initial_speed_mps - traj.min_speed_mps
            min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
            if speed_drop >= 0.2 and 3.0 <= min_speed_time <= 5.0:  # Timing window for speed drop
                comp["decelerate_commitment_execution"] = 0.7 * min(1.0, speed_drop / 10.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
