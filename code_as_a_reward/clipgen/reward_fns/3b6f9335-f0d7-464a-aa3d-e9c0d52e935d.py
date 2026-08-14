"""clip 3b6f9335-f0d7-464a-aa3d-e9c0d52e935d - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive events:
    1. Stopping at the red traffic light behind the lead vehicle.
       - Perceptual mention: {'vehicle_generic', 'lead_vehicle', 'signal'}
       - Commitment: speed_profile='decelerate'
       - Trajectory: Speed drop to approximately 0.0 m/s, with a minimum floor of 0.05 m/s drop, and timing around t=0.5s.
    """
    # Initialize component scores
    comp = {
        "perceptual_vehicle_or_signal": 0.0,
        "commitment_decelerate": 0.0,
    }

    # Perceptual mention of vehicle or signal
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'signal') for p in claims.perceptual):
        comp["perceptual_vehicle_or_signal"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded trajectory factor for deceleration with timing consideration
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        min_speed_time = min_speed_time_idx * traj.dt_s

        # Ensure the minimum speed occurs around the expected time
        if 0.0 <= min_speed_time <= 1.0:  # Allow some tolerance around the expected time
            deceleration_factor = 0.6 * min(1.0, speed_drop / 0.05)  # GT drop is 0.1 m/s
            comp["commitment_decelerate"] = deceleration_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
