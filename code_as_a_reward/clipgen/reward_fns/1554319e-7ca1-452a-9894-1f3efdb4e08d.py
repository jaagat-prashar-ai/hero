"""clip 1554319e-7ca1-452a-9894-1f3efdb4e08d - attempt 2/5 - gate PASS (pos 1.00, max pert 0.20, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene 1554319e-7ca1-452a-9894-1f3efdb4e08d:
    - Decelerate due to snowy conditions.
    - Trajectory should show a speed drop of at least 2.1 m/s within the first 4.5 seconds.
    """
    comp = {
        "decelerate_commitment": 0.0,
        "trajectory_deceleration": 0.0
    }

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.2

        # Trajectory deceleration factor
        initial_speed = traj.initial_speed_mps
        min_speed_after = traj.min_speed_mps
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop, expecting at least 2.1 m/s drop
        if speed_drop > 2.1:
            # Ensure the deceleration occurs within the first 4.5 seconds
            min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 4.5))
            min_speed_time = min_speed_time_idx * traj.dt_s
            if min_speed_time <= 4.5:
                comp["trajectory_deceleration"] = 0.8 * min(1.0, speed_drop / 4.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
