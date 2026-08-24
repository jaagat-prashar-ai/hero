"""clip 202c7cf3-0f56-4abf-ae21-5c3efbd7a323 - attempt 2/5 - gate PASS (pos 0.75, max pert 0.05, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene 202c7cf3-0f56-4abf-ae21-5c3efbd7a323:
    - Stop at the stop line due to the stop sign and pedestrians crossing.
    - Thresholds: speed drop >= 1.0 m/s by t=5.5 s.
    """
    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.05,
        "commitment_decelerate": 0.0,
        "trajectory_slowing": 0.0,
    }

    # Perceptual claim: pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.05

    # Commitment claim: decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory: slowing
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, 5.5))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for slowing
        if speed_drop >= 1.0:
            comp["commitment_decelerate"] = 0.2
            comp["trajectory_slowing"] = 0.5 * min(1.0, speed_drop / 1.9)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
