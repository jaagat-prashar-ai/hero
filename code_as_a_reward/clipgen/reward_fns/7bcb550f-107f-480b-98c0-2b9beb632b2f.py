"""clip 7bcb550f-107f-480b-98c0-2b9beb632b2f - attempt 1/5 - gate PASS (pos 0.90, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrian yielding and vehicle interaction.
    
    Decisive Events:
    1. Interaction with Track 216 [automobile]: Deceleration due to close vehicle.
       - Perceptual mention: vehicle_generic
       - Commitment: decelerate
       - Trajectory: Speed drop of at least 1.0 m/s after t=2.1s
    
    2. Yielding to Pedestrian: Deceleration to yield to pedestrian.
       - Perceptual mention: pedestrian
       - Commitment: decelerate
       - Trajectory: Speed drop of at least 1.0 m/s throughout the window
    """
    # Initialize component scores
    comp = {
        "mention_vehicle": 0.0,
        "mention_pedestrian": 0.0,
        "decelerate_for_vehicle": 0.0,
        "decelerate_for_pedestrian": 0.0
    }
    
    # Check perceptual mentions
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        comp["mention_vehicle"] = 0.1

    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for deceleration commitment
    decelerate_claim = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Trajectory analysis
    if traj.n_waypoints > 0:
        # Speed drop for vehicle interaction
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after_vehicle = np.min(window(speed_series, traj.dt_s, 2.1, traj.dt_s * traj.n_waypoints))
        speed_drop_vehicle = initial_speed - min_speed_after_vehicle
        if decelerate_claim:
            comp["decelerate_for_vehicle"] = 0.4 * min(1.0, speed_drop_vehicle / 2.0)

        # Speed drop for pedestrian yielding
        min_speed_overall = traj.min_speed_mps
        speed_drop_pedestrian = initial_speed - min_speed_overall
        if decelerate_claim:
            comp["decelerate_for_pedestrian"] = 0.4 * min(1.0, speed_drop_pedestrian / 2.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
