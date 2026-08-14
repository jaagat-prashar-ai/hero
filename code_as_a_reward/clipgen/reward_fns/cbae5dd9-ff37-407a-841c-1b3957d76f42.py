"""clip cbae5dd9-ff37-407a-841c-1b3957d76f42 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene cbae5dd9-ff37-407a-841c-1b3957d76f42:
    - Yield to pedestrian at crosswalk: perceptual mention of pedestrian or crosswalk,
      commitment to decelerate, trajectory speed drop of at least 0.9 m/s at the correct time.
    """
    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_decelerate": 0.0,
    }

    # Perceptual mention of pedestrian or crosswalk
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["commitment_decelerate"] = 0.3

        # Trajectory speed drop with timing consideration
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
        if speed_drop >= 0.9 and 1.5 <= min_speed_time <= 3.5:
            comp["trajectory_decelerate"] = 0.6 * min(1.0, speed_drop / 1.8)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
