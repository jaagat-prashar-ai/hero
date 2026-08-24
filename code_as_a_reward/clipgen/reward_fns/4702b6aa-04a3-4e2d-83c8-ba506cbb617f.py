"""clip 4702b6aa-04a3-4e2d-83c8-ba506cbb617f - attempt 1/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scene 4702b6aa-04a3-4e2d-83c8-ba506cbb617f:
    - Decelerate to maintain safe distance from lead vehicle and adjacent lane vehicle.
    - Trajectory thresholds: small speed drop (~0.5 m/s) around t=4.1s and t=1.3s.
    - Perceptual credit for mentioning 'lead_vehicle' or 'vehicle_generic'.
    """

    # Initialize component scores
    comp = {
        "perceptual_lead_vehicle": 0.0,
        "perceptual_adjacent_vehicle": 0.0,
        "decelerate_lead_vehicle": 0.0,
        "decelerate_adjacent_vehicle": 0.0,
    }

    # Check for perceptual claims
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["perceptual_lead_vehicle"] = 0.05

    if any(p.entity in ('vehicle_generic', 'lead_vehicle') for p in claims.perceptual):
        comp["perceptual_adjacent_vehicle"] = 0.05

    # Check for deceleration commitment
    decelerate_claim = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Trajectory analysis for lead vehicle
    if decelerate_claim:
        speed_window = window(traj.speed_mps, traj.dt_s, 0.0, 6.4)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(speed_window)
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop
        if speed_drop > 0.25:
            comp["decelerate_lead_vehicle"] = 0.5 * min(1.0, speed_drop / 0.5)

    # Trajectory analysis for adjacent lane vehicle
    if decelerate_claim:
        speed_window = window(traj.speed_mps, traj.dt_s, 0.0, 6.4)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(speed_window)
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop
        if speed_drop > 0.25:
            comp["decelerate_adjacent_vehicle"] = 0.4 * min(1.0, speed_drop / 0.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
