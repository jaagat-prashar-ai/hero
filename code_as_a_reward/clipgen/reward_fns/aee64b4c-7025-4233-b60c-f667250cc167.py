"""clip aee64b4c-7025-4233-b60c-f667250cc167 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 8)"""
def components(claims, traj):
    """Components for scene aee64b4c-7025-4233-b60c-f667250cc167:
    - Deceleration to maintain a safe distance from the lead vehicle.
    - Perceptual mention of the lead vehicle.
    - Trajectory should show a speed drop of at least 0.5 m/s around t=3.2s to t=4.3s.
    """
    perceptual_credit = 0.0
    deceleration_credit = 0.0

    # Check for perceptual mention of the lead vehicle
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_credit = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop during the relevant time window
        speed_window = window(traj.speed_mps, traj.dt_s, 3.2, 4.3)
        if len(speed_window) > 0:
            min_speed_after = np.min(speed_window)
            speed_drop = traj.initial_speed_mps - min_speed_after
            # Graded trajectory factor for deceleration
            deceleration_credit = 0.6 * min(1.0, speed_drop / 1.0)

    return {
        "perceptual_mention": perceptual_credit,
        "deceleration_execution": deceleration_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
