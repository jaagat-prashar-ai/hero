"""clip b2320fdb-d7b4-49d9-be9e-09b6f70e6f7f - attempt 2/5 - gate PASS (pos 0.90, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for the scene where the expert adapts speed for road curvature
    and navigates through a construction zone. The decisive events are:
    1. Adapting speed for road curvature: Expect a speed profile of 'decelerate'
       or 'maintain', with a trajectory showing a significant speed change.
    Trajectory thresholds are set to approximately half the GT magnitudes.
    """

    # Initialize component scores
    comp = {
        "perceptual_mention": 0.0,
        "speed_adaptation": 0.0,
    }

    # Perceptual mention credit
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'curve') for p in claims.perceptual):
        comp["perceptual_mention"] = 0.1

    # Speed adaptation commitment and trajectory
    if any(c.speed_profile in ('decelerate', 'maintain') for c in claims.commitments):
        # Calculate speed change
        speed_change = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed adaptation with timing consideration
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        min_speed_time = min_speed_time_idx * traj.dt_s
        if min_speed_time >= 3.0:  # Ensure the minimum speed occurs later in the window
            comp["speed_adaptation"] = 0.9 * min(1.0, speed_change / 2.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
