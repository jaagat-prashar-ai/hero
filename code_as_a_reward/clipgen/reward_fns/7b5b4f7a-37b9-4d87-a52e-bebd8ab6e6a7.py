"""clip 7b5b4f7a-37b9-4d87-a52e-bebd8ab6e6a7 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.38, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scene with construction zone and nearby vehicles.
    
    Decisive Events:
    1. Construction Zone Ahead: Maintain low speed due to pedestrians and construction equipment.
    2. Proximity of Nearby Vehicles: Maintain lane position without lateral movement.
    
    Scene-derived thresholds:
    - Speed: Maintain below 1.0 m/s for construction zone, full credit below 0.5 m/s.
    - Lateral Offset: Maintain within ±0.5 m.
    """
    comp = {
        "perceptual_construction_zone": 0.0,
        "commitment_slowing": 0.0,
        "trajectory_slowing": 0.0,
        "perceptual_nearby_vehicles": 0.0,
        "trajectory_lateral_stability": 0.0
    }

    # Perceptual credit for construction zone
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction_zone"] = 0.1

    # Commitment and trajectory for slowing
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for slowing
        comp["trajectory_slowing"] = 0.5 * min(1.0, speed_drop / 0.5)

        # Combine with commitment
        comp["commitment_slowing"] = 0.3 if speed_drop >= 0.5 else 0.0

    # Perceptual credit for nearby vehicles
    if any(p.entity in ('vehicle_generic', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        comp["perceptual_nearby_vehicles"] = 0.1

    # Trajectory for lateral stability
    lateral_offsets = window(traj.lateral_offset_m, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s)
    max_lateral_deviation = max(abs(lateral_offsets))
    comp["trajectory_lateral_stability"] = 0.1 if max_lateral_deviation <= 0.5 else 0.0

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
