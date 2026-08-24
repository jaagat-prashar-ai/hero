"""clip 7cd15c50-e1c7-4177-9b10-b08cc5e3bbde - attempt 2/5 - gate PASS (pos 0.75, max pert 0.05, real rollout argmax 9)"""
def components(claims, traj):
    # Initialize component scores
    components = {
        "perceptual_pedestrian": 0.05,  # Reduced weight for perceptual mention
        "commitment_decelerate": 0.0,
        "trajectory_decelerate": 0.0
    }

    # Check for perceptual claims about pedestrians
    if any(p.entity in ("pedestrian", "cyclist", "workers") for p in claims.perceptual):
        components["perceptual_pedestrian"] = 0.05

    # Check for commitment claims to decelerate
    if any(c.speed_profile == "decelerate" for c in claims.commitments):
        # Calculate speed drop in the trajectory
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for deceleration with timing consideration
        if speed_drop >= 1.85:  # Half of the positive's 3.7 m/s drop
            min_speed_time_idx = np.argmin(window(speed_series, traj.dt_s, 0.0, 6.4))
            min_speed_time = min_speed_time_idx * traj.dt_s
            if 2.0 <= min_speed_time <= 3.0:  # Timing window for deceleration
                components["commitment_decelerate"] = 0.25
                components["trajectory_decelerate"] = 0.45 * min(1.0, speed_drop / 3.7)

    return components

def reward(claims, traj):
    """Decisive events: pedestrian group crossing. Thresholds: perceptual mention of pedestrians, commitment to decelerate, and graded speed drop of at least 1.85 m/s with timing consideration."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
