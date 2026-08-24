"""clip 8d5b01da-99ee-4f7c-aeb0-e5bef0ae9d3d - attempt 4/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with stopping behind a lead vehicle.
    
    Decisive Events:
    1. Stopping for the lead vehicle: Expect a mention of a vehicle entity and a deceleration commitment.
    
    Scene-derived thresholds:
    - Speed drop for stopping: at least 0.45 m/s (half of 0.9 m/s).
    """
    # Initialize component scores
    component_scores = {
        "mention_vehicle": 0.0,
        "decelerate_commitment": 0.0,
    }

    # Perceptual mentions
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        component_scores["mention_vehicle"] = 0.1

    # Commitment and trajectory for stopping
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Ensure the speed drop is significant and occurs early in the trajectory
        if speed_drop >= 0.45 and np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s <= 2.3:
            component_scores["decelerate_commitment"] = 0.9 * min(1.0, speed_drop / 0.9)

    return component_scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
