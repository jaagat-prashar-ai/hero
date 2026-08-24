"""clip d00e6d02-82ee-4430-a3f6-dacf408aa842 - attempt 3/5 - gate PASS (pos 0.75, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Cyclist presence and deceleration: Expect a speed drop of at least 0.4 m/s.
    2. Pedestrian presence: Minor speed drop acknowledgment.
    Trajectory factors are graded and one-sided, with commitment credit matched at the FAMILY level.
    """
    # Initialize component scores
    comp = {
        "mention_cyclist": 0.0,
        "decelerate_for_cyclist": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity in ('cyclist', 'vehicle_generic') for p in claims.perceptual):
        comp["mention_cyclist"] = 0.05

    # Check for deceleration commitment
    decelerate_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Calculate speed drop
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps

    # Deceleration for cyclist
    if decelerate_commitment and speed_drop >= 0.4:
        # Ensure the minimum speed occurs early in the window
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints)) * traj.dt_s
        if min_speed_time <= 2.0:
            comp["decelerate_for_cyclist"] = 0.5 * min(1.0, speed_drop / 3.2)

    # Deceleration for pedestrians
    if decelerate_commitment and speed_drop >= 0.2:
        # Ensure the minimum speed occurs early in the window
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints)) * traj.dt_s
        if min_speed_time <= 2.0:
            comp["decelerate_for_pedestrian"] = 0.2 * min(1.0, speed_drop / 3.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
