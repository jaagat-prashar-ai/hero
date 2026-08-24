"""clip d05dcfff-5aee-4402-85fa-2cc1cdbd2545 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.15, real rollout argmax 0)"""
def components(claims, traj):
    """Components for the scene where the expert yields to pedestrians at a crosswalk.
    Decisive events:
    - Yield to pedestrians: Expect a 'decelerate' commitment and a speed drop of at least 0.6 m/s.
    - Perceptual mention of 'pedestrian' or 'crosswalk'.
    Trajectory thresholds are derived from the ground truth: speed drop of 1.2 m/s, with a graded floor at 0.6 m/s.
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "perceptual_crosswalk": 0.0,
        "yield_execution": 0.0,
    }

    # Perceptual mention credit
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Commitment and trajectory execution credit
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded execution credit for yielding
        comp["yield_execution"] = 0.6 * min(1.0, speed_drop / 1.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
