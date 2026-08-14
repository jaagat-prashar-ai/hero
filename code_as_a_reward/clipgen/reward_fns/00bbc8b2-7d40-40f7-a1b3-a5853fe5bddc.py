"""clip 00bbc8b2-7d40-40f7-a1b3-a5853fe5bddc - attempt 2/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive events:
    1. Following the lead vehicle: Expect mention of 'lead_vehicle' and maintaining speed.
    2. Approaching the construction zone: Expect mention of 'work_zone' but no specific commitment.
    
    Trajectory thresholds:
    - Speed maintenance: Minimal speed change, floor at 0.05 m/s.
    """

    # Initialize component scores
    comp = {
        "mention_lead_vehicle": 0.0,
        "maintain_speed": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["mention_lead_vehicle"] = 0.1

    # Check for speed maintenance commitment
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        # Calculate speed change and timing
        speed_change = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        min_speed_time = min_speed_time_idx * traj.dt_s

        # Graded factor for maintaining speed, considering timing
        if min_speed_time > 3.0:  # Ensure the speed drop occurs later in the window
            comp["maintain_speed"] = 0.7 * max(0.0, min(1.0, speed_change / 0.3))

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
