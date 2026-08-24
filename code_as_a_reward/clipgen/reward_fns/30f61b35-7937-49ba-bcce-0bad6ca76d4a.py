"""clip 30f61b35-7937-49ba-bcce-0bad6ca76d4a - attempt 3/5 - gate PASS (pos 0.70, max pert 0.11, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with a pedestrian crossing, requiring deceleration."""
    comp = {
        "saw_pedestrian": 0.0,
        "decelerate_executed": 0.0,
    }

    # Perceptual component: mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.1

    # Commitment component: deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the window
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Adjusted graded factor for deceleration execution
        # Consider the timing of the minimum speed
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        min_speed_time = min_speed_time_idx * traj.dt_s

        if min_speed_time <= 5.4:
            comp["decelerate_executed"] = 0.6 * min(1.0, speed_drop / 3.9)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
