"""clip 4a3d5c8f-36f1-4106-b85b-0506f3a0b78e - attempt 5/5 - gate PASS (pos 0.90, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive events:
    1. Pedestrian proximity: Expect mention of 'pedestrian' and a deceleration.
    Trajectory thresholds are set to half the ground truth's magnitude for speed drop.
    """

    # Initialize component scores
    comp = {
        "saw_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }

    # Perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.1

    # Trajectory analysis
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))

    # Commitment claims and trajectory conjunctions
    if any(c.speed_profile == 'decelerate' for c in claims.commitments) and speed_drop >= 0.95 and min_speed_time > 3.0:
        deceleration_factor = 0.8 * min(1.0, speed_drop / 0.95)  # Graded factor with a floor at 0.95 m/s drop
        comp["decelerate_for_pedestrian"] = deceleration_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
