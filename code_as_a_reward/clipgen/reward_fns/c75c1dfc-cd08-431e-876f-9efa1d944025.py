"""clip c75c1dfc-cd08-431e-876f-9efa1d944025 - attempt 2/5 - gate PASS (pos 0.91, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene c75c1dfc-cd08-431e-876f-9efa1d944025:
    - Yield to crossing pedestrians: perceptual mention of 'pedestrian',
      commitment to 'decelerate', speed drop of at least 0.85 m/s, graded
      factor based on speed drop, and timing of minimum speed.
    """
    comp = {
        "perceptual_pedestrian": 0.0,
        "yield_execution": 0.0
    }

    # Perceptual mention of pedestrians
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Yield to pedestrians: decelerate commitment and speed drop with timing
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
        if speed_drop >= 0.85 and 1.5 <= min_speed_time <= 2.5:
            comp["yield_execution"] = 0.9 * min(1.0, speed_drop / 1.7)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
