"""clip 83bac332-fbd6-45f9-85ef-ec940ff3422f - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 9)"""
def components(claims, traj):
    """
    Components for scene 83bac332-fbd6-45f9-85ef-ec940ff3422f:
    - Deceleration to yield to a pedestrian crossing the road.
    - Expect a perceptual mention of 'pedestrian' and a commitment to 'decelerate'.
    - Trajectory should show a deceleration of at least 0.5 m/s, with timing considered.
    """
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after_t3 = np.min(window(traj.speed_mps, traj.dt_s, 3.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after_t3

        # Graded factor for deceleration, considering timing
        if speed_drop >= 0.5:
            comp["decelerate_for_pedestrian"] = 0.6 * min(1.0, speed_drop / 0.9)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
