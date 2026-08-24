"""clip c7879708-a0ed-40cb-a2aa-d122140076ee - attempt 2/5 - gate PASS (pos 0.99, max pert 0.05, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene c7879708-a0ed-40cb-a2aa-d122140076ee:
    Deceleration to maintain a safe distance from the lead vehicle.
    - Perceptual: any of 'lead_vehicle', 'vehicle_generic'
    - Commitment: speed_profile='decelerate'
    - Trajectory: deceleration of at least 2.2 m/s (half of 4.4 m/s drop) early in the window
    """
    perceptual_weight = 0.05
    commitment_weight = 0.65
    trajectory_weight = 0.30

    # Perceptual component
    saw_lead_vehicle = any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual)
    perceptual_score = perceptual_weight if saw_lead_vehicle else 0.0

    # Commitment component
    committed_to_decelerate = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    speed_series = np.array(traj.speed_mps)
    initial_speed = traj.initial_speed_mps
    min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, 6.4))
    speed_drop = initial_speed - min_speed_after
    trajectory_score = 0.0

    if committed_to_decelerate:
        # Graded trajectory factor for deceleration
        if speed_drop >= 2.2:
            trajectory_score = trajectory_weight * min(1.0, speed_drop / 4.4)
        commitment_score = commitment_weight * min(1.0, speed_drop / 4.4)
    else:
        commitment_score = 0.0

    return {
        "perceptual": perceptual_score,
        "commitment": commitment_score,
        "trajectory": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
