"""clip dcf31ac5-7733-4075-b178-2ff24f604787 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.30, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing and deceleration:
    - Perceptual mention of pedestrian
    - Commitment to decelerate (stop/yield/wait/decelerate family)
    - Trajectory showing speed drop of at least 1.3 m/s, graded, with timing
    """
    comp = {
        "perceptual_pedestrian": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_deceleration": 0.0,
    }

    # Perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["commitment_decelerate"] = 0.2

        # Trajectory showing deceleration with timing consideration
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Find the time of minimum speed
        min_speed_idx = np.argmin(window(speed_series, traj.dt_s, 0.0, 6.4))
        min_speed_time = min_speed_idx * traj.dt_s

        # Check for speed drop and timing
        if speed_drop >= 1.3 and 1.0 <= min_speed_time <= 3.0:
            comp["trajectory_deceleration"] = 0.5 * min(1.0, speed_drop / 2.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
