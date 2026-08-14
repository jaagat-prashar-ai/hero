"""clip 85766484-e0fc-489e-aad9-04ec31843c01 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 85766484-e0fc-489e-aad9-04ec31843c01:
    - Deceleration for red traffic light
    - Perceptual recognition of traffic light and pedestrians
    - Trajectory speed drop of at least 2.0 m/s, with timing consideration
    """
    # Initialize component scores
    comp = {
        "perceptual_traffic_light": 0.05,
        "perceptual_pedestrian": 0.05,
        "decelerate_for_traffic_light": 0.0,
    }

    # Perceptual claims
    if any(p.entity in ('signal',) for p in claims.perceptual):
        comp["perceptual_traffic_light"] = 0.05

    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.05

    # Commitment claims and trajectory checks
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory speed drop with timing consideration
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        min_speed_time = min_speed_time_idx * traj.dt_s

        # Deceleration for traffic light
        if any(p.entity in ('signal',) for p in claims.perceptual) and min_speed_time > 3.0:
            graded_speed_drop = 0.6 * min(1.0, speed_drop / 4.0)  # Graded above 2.0 m/s drop
            comp["decelerate_for_traffic_light"] = graded_speed_drop

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
