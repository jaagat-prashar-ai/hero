"""clip f7ae75be-a19f-4aaa-9fc7-a7297519284b - attempt 2/5 - gate PASS (pos 1.00, max pert 0.05, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Yield to Lead Vehicle: Expect a speed drop with a 'decelerate' commitment.
    2. Maintain Position for Construction Cones: Expect low speed with a 'decelerate' commitment.
    Trajectory thresholds are derived from the scene's GT trajectory, with graded factors.
    """

    # Initialize component scores
    comp = {
        "perceptual_vehicle": 0.0,
        "yield_execution": 0.0,
        "maintain_position_execution": 0.0
    }

    # Perceptual credit for mentioning vehicle-related entities
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.05

    # Commitment and trajectory check for yielding to lead vehicle
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded factor for speed drop, focusing on the timing of the minimum speed
        if traj.n_waypoints > 0:
            min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
            min_speed_time = min_speed_time_idx * traj.dt_s
            if min_speed_time >= 6.0:  # Ensure the minimum speed occurs late in the window
                comp["yield_execution"] = 0.45 * min(1.0, speed_drop / 0.1)

    # Commitment and trajectory check for maintaining position for construction cones
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate max speed
        max_speed = max(traj.speed_mps)

        # Graded factor for maintaining low speed
        comp["maintain_position_execution"] = 0.5 * min(1.0, (0.6 - max_speed) / 0.3)

    return comp

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
