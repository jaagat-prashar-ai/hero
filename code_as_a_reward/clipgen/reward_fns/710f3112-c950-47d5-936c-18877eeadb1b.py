"""clip 710f3112-c950-47d5-936c-18877eeadb1b - attempt 2/5 - gate PASS (pos 0.97, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene 710f3112-c950-47d5-936c-18877eeadb1b:
    - Deceleration in response to a pedestrian at a crosswalk.
    - Thresholds: speed drop >= 3.3 m/s, focus on timing of deceleration.
    """
    comp = {
        "mention_pedestrian": 0.0,
        "mention_crosswalk": 0.0,
        "decelerate_executed": 0.0
    }

    # Perceptual mentions
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05
        comp["mention_crosswalk"] = 0.05

    # Deceleration commitment and execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Focus on timing: ensure deceleration occurs early enough
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        min_speed_time = min_speed_time_idx * traj.dt_s
        if speed_drop >= 3.3 and min_speed_time <= 4.4:
            comp["decelerate_executed"] = 0.9 * min(1.0, speed_drop / 6.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
