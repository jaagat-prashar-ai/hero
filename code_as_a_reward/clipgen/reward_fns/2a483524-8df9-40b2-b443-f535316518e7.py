"""clip 2a483524-8df9-40b2-b443-f535316518e7 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for Clip 2a483524-8df9-40b2-b443-f535316518e7:
    - Decisive Event 1: Pedestrian crossing at the crosswalk, requiring deceleration.
      Thresholds: speed drop >= 0.75 m/s, primarily in the first half of the window.
    - Nearby automobiles are background; no specific maneuver required.
    """
    comp = {
        "saw_pedestrian": 0.0,
        "saw_crosswalk": 0.0,
        "decelerate_executed": 0.0,
    }

    # Perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.05

    if any(p.entity == 'crosswalk' for p in claims.perceptual):
        comp["saw_crosswalk"] = 0.05

    # Commitment claims and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop within the window
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed = np.min(window(speed_series, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed

        # Graded factor for deceleration
        if speed_drop >= 0.75:
            comp["decelerate_executed"] = 0.6 * min(1.0, speed_drop / 1.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
