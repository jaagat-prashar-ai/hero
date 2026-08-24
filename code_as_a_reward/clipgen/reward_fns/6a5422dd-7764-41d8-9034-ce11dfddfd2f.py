"""clip 6a5422dd-7764-41d8-9034-ce11dfddfd2f - attempt 2/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 6a5422dd-7764-41d8-9034-ce11dfddfd2f:
    - Yield to pedestrian crossing the road: perceptual mention of pedestrian,
      commitment to decelerate, and trajectory showing speed reduction.
    - Trajectory thresholds: speed drop >= 2.0 m/s, graded factor for deceleration.
    """
    result = {
        "mention_pedestrian": 0.0,
        "decelerate_commitment_execution": 0.0
    }

    # Perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        result["mention_pedestrian"] = 0.1

    # Commitment to decelerate with matching trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_window = window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(speed_window)
        speed_drop = initial_speed - min_speed_after

        if speed_drop >= 2.0:
            result["decelerate_commitment_execution"] = 0.7 * min(1.0, speed_drop / 4.0)

    return result

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
