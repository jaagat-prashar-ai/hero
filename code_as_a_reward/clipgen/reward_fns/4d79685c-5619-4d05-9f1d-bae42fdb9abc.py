"""clip 4d79685c-5619-4d05-9f1d-bae42fdb9abc - attempt 3/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive events:
    1. Stopping for pedestrians at the crosswalk.
    2. Yielding for a vehicle turning into the lane.
    
    Trajectory thresholds are derived from the ground-truth dossier:
    - Speed drop of at least 0.6 m/s (half of the 1.2 m/s drop in GT).
    - Timing for speed drop around t=2.9 s.
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "perceptual_vehicle": 0.0,
        "stop_for_pedestrian": 0.0,
        "yield_for_vehicle": 0.0,
    }

    # Perceptual mention-only credit
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.05

    if any(p.entity in ('vehicle_generic', 'lane') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.05

    # Check for commitment claims and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = traj.min_speed_mps
        speed_drop = initial_speed - min_speed_after

        # Timing of minimum speed
        min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))

        # Graded factor for stopping for pedestrians
        if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual) and min_speed_time <= 3.0:
            comp["stop_for_pedestrian"] = 0.45 * min(1.0, speed_drop / 1.2)

        # Graded factor for yielding for vehicles
        if any(p.entity in ('vehicle_generic', 'lane') for p in claims.perceptual) and min_speed_time <= 3.0:
            comp["yield_for_vehicle"] = 0.45 * min(1.0, speed_drop / 1.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
