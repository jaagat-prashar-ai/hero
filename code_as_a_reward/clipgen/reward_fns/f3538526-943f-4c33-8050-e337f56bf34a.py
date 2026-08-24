"""clip f3538526-943f-4c33-8050-e337f56bf34a - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with oncoming vehicle and rider, requiring deceleration and minimal movement."""
    comp = {
        "decelerate_execution": 0.0,
    }

    # Commitment and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop and ensure it occurs at the correct time
        speed_series = np.array(traj.speed_mps)
        min_speed_idx = np.argmin(window(speed_series, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        min_speed_time = min_speed_idx * traj.dt_s

        # Ensure the speed drop is significant and occurs later in the window
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if min_speed_time >= 3.0:  # Ensure the drop occurs later in the window
            comp["decelerate_execution"] = 0.7 * min(1.0, speed_drop / 0.75)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
