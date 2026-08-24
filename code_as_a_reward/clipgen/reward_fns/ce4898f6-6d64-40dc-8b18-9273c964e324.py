"""clip ce4898f6-6d64-40dc-8b18-9273c964e324 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene ce4898f6-6d64-40dc-8b18-9273c964e324:
    - Deceleration for a red traffic light: perceptual mention of 'signal',
      commitment to 'decelerate', and a speed drop of at least 1.65 m/s
      within the window.
    - Perceptual mention credit is small and independent.
    """
    comp = {
        "perceptual_signal": 0.0,
        "decelerate_commitment": 0.0,
        "decelerate_execution": 0.0,
    }

    # Perceptual mention of a traffic signal
    if any(p.entity == 'signal' for p in claims.perceptual):
        comp["perceptual_signal"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.2

        # Trajectory execution: speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints))
        speed_drop = initial_speed - min_speed_after

        # Graded execution score based on speed drop
        if speed_drop >= 1.65:
            comp["decelerate_execution"] = 0.5 * min(1.0, speed_drop / 3.3)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
