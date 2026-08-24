"""clip 1f74d6b9-029a-4a09-b4bb-2b63b1998f2b - attempt 2/5 - gate PASS (pos 0.95, max pert 0.05, real rollout argmax 9)"""
def components(claims, traj):
    """Components for scene 1f74d6b9-029a-4a09-b4bb-2b63b1998f2b:
    - Maintain speed while keeping a safe distance from the lead vehicle.
    - Monitor the heavy truck in the right adjacent lane.
    - Trajectory thresholds: speed drop >= 1.0 m/s, with timing consideration.
    """

    # Initialize component scores
    comp = {
        "mention_lead_vehicle": 0.0,
        "mention_truck": 0.0,
        "maintain_speed": 0.0,
    }

    # Perceptual mentions
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["mention_lead_vehicle"] = 0.05

    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        comp["mention_truck"] = 0.05

    # Commitment and trajectory for maintaining speed
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
        if speed_drop >= 1.0 and min_speed_time >= 3.0:
            comp["maintain_speed"] = 0.9 * min(1.0, speed_drop / 2.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
