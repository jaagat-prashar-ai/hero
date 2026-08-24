"""clip d82a4d30-d930-4518-874e-0bc48dfe1744 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scene d82a4d30-d930-4518-874e-0bc48dfe1744:
    - Deceleration in response to the lead bus (track 165).
    - Perceptual mention of 'lead_vehicle' or 'vehicle_generic'.
    - Trajectory speed drop of at least 1.5 m/s by t=5.5 s.
    """
    comp = {
        "perceptual_lead_vehicle": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_speed_drop": 0.0,
    }

    # Perceptual mention of lead vehicle or generic vehicle
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["perceptual_lead_vehicle"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for speed drop
        comp["trajectory_speed_drop"] = 0.5 * min(1.0, speed_drop / 2.9)

        # Combine commitment and trajectory for deceleration
        comp["commitment_decelerate"] = 0.4 * min(1.0, speed_drop / 1.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
