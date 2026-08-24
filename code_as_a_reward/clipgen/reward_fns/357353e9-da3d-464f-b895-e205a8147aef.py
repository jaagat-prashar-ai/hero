"""clip 357353e9-da3d-464f-b895-e205a8147aef - attempt 4/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 2)"""
def components(claims, traj):
    """Decisive events: yield to pedestrians, maintain safe distance from lead vehicle.
    Trajectory thresholds: speed drop >= 1.55 m/s, graded above this floor.
    Perceptual mentions: 'pedestrian'.
    Commitment families: 'decelerate' for slowing.
    """
    # Initialize component scores
    comp = {
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for deceleration commitment and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.1, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Find the time of minimum speed
        min_speed_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0.1, traj.n_waypoints * traj.dt_s))
        min_speed_time = 0.1 + min_speed_idx * traj.dt_s

        # Deceleration for pedestrians
        if any(p.entity == 'pedestrian' for p in claims.perceptual) and min_speed_time >= 3.0:
            comp["decelerate_for_pedestrian"] = 0.70 * min(1.0, speed_drop / 3.1)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
