"""clip b786ada1-87d8-4f1e-8d1a-4178d05410c2 - attempt 5/5 - gate PASS (pos 0.80, max pert 0.00, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene b786ada1-87d8-4f1e-8d1a-4178d05410c2:
    - Yield to pedestrian crossing: decelerate with speed drop >= 0.9 m/s
    - Stop behind motorcycle: decelerate with speed drop >= 0.9 m/s
    Trajectory thresholds are graded and one-sided, with a focus on speed drop.
    """
    # Initialize component scores
    comp = {
        "decelerate_for_pedestrian": 0.0,
        "decelerate_for_motorcycle": 0.0
    }
    
    # Trajectory and commitment checks
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s

    # Decelerate for pedestrian
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        if speed_drop >= 0.9 and min_speed_time <= 1.5:
            comp["decelerate_for_pedestrian"] = 0.5 * min(1.0, speed_drop / 1.8)

    # Decelerate for motorcycle
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        if speed_drop >= 0.9 and min_speed_time <= 1.5:
            comp["decelerate_for_motorcycle"] = 0.5 * min(1.0, speed_drop / 1.8)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
