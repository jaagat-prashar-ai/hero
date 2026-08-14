"""clip bcc52cd1-9507-425e-a5a6-5f8a795ed525 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of deceleration due to a vehicle turning into the lane.
    Thresholds derived from the expert trajectory: speed drop of 1.4 m/s, minimum speed at t=2.1s.
    """

    # Initialize component scores
    perceptual_vehicle = 0.0
    deceleration_executed = 0.0

    # Perceptual component: mention of vehicle-related entity
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        perceptual_vehicle = 0.1

    # Commitment and trajectory component: deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for deceleration
        deceleration_executed = 0.6 * min(1.0, speed_drop / 1.4)

    return {
        "perceptual_vehicle": perceptual_vehicle,
        "deceleration_executed": deceleration_executed,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
