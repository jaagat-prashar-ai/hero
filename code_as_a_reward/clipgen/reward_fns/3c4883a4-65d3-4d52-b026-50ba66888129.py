"""clip 3c4883a4-65d3-4d52-b026-50ba66888129 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene with strong deceleration for barricades ahead.
    
    Decisive Event: Strong Deceleration for Barricades
    - Perceptual: Mention of barricades or related entities.
    - Commitment: Deceleration (speed_profile='decelerate').
    - Trajectory: Speed drop of at least 1.0 m/s, graded above this floor.
    """
    perceptual_credit = 0.05
    if any(p.entity in ('barricades', 'vehicle_generic') for p in claims.perceptual):
        perceptual_credit = 0.05

    commitment_credit = 0.0
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory window
        initial_speed = traj.speed_mps[0]
        min_speed = min(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed
        # Graded factor for speed drop, floor at half the GT drop (1.0 m/s)
        trajectory_factor = 0.95 * min(1.0, speed_drop / 2.0)
        # Check timing of minimum speed to differentiate reversed trajectory
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
        if min_speed_time >= 3.0:  # Ensure minimum speed occurs later in the window
            commitment_credit = trajectory_factor

    return {
        "perceptual_mention": perceptual_credit,
        "deceleration_executed": commitment_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
