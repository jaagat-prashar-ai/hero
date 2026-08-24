"""clip 716976eb-6e32-4b31-952d-793b7dbc5ae1 - attempt 1/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing and nearby vehicles.
    
    Decisive Events:
    1. Pedestrian Crossing: Requires deceleration.
       - Perceptual mention: 'pedestrian', 'crosswalk'
       - Commitment: speed_profile='decelerate'
       - Trajectory: Speed drop >= 3.65 m/s after 4.1 s
    2. Nearby Automobiles: Monitored but no specific maneuver required.
       - Perceptual mention: 'vehicle_generic'
       - No specific commitment or trajectory change required.
    
    Trajectory thresholds are graded and one-sided, with generous floors.
    """

    # Initialize component scores
    comp = {
        "saw_pedestrian": 0.0,
        "saw_crosswalk": 0.0,
        "decelerate_executed": 0.0,
        "saw_vehicle": 0.0,
    }

    # Perceptual claims
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.05
        comp["saw_crosswalk"] = 0.05

    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        comp["saw_vehicle"] = 0.05

    # Commitment claims and trajectory checks
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop after pedestrian becomes visible
        speed_window = window(traj.speed_mps, traj.dt_s, 4.1, traj.n_waypoints * traj.dt_s)
        if len(speed_window) > 0:
            min_speed_after = np.min(speed_window)
            speed_drop = traj.initial_speed_mps - min_speed_after
            # Graded trajectory factor
            comp["decelerate_executed"] = 0.7 * min(1.0, speed_drop / 6.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
