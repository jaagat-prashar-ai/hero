"""clip c41365fe-0371-4ebc-81bd-6a6a3494cb73 - attempt 2/5 - gate PASS (pos 0.90, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for reward calculation based on decisive events:
    1. Initial Stop and Creep Forward: Expect a 'decelerate' commitment with a speed drop of at least 2.6 m/s by t=6.3 s, followed by a gradual speed increase.
    """
    comp = {
        "commitment_decelerate": 0.0,
        "trajectory_stop_and_creep": 0.0,
    }

    # Commitment component: decelerate family
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory component: stop and creep forward
        speed_series = np.array(traj.speed_mps)
        min_speed_idx = np.argmin(window(speed_series, traj.dt_s, 0.0, 6.4))
        min_speed_time = min_speed_idx * traj.dt_s

        # Speed drop and increase
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        final_speed = traj.final_speed_mps

        speed_drop = initial_speed - min_speed

        if min_speed_time >= 6.3:
            # Graded factor for speed drop
            drop_factor = 0.7 * min(1.0, speed_drop / 5.2)
            comp["commitment_decelerate"] = 0.2
            comp["trajectory_stop_and_creep"] = drop_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
