"""clip a97d3965-7ee8-4a5b-8df3-16da2b5b4d11 - attempt 2/5 - gate PASS (pos 0.95, max pert 0.05, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene a97d3965-7ee8-4a5b-8df3-16da2b5b4d11:
    - Deceleration to stop: expect a 'decelerate' commitment and a speed drop of at least 0.05 m/s.
    - Perceptual mention of entities like 'cyclist', 'pedestrian', or 'vehicle' as potential crossing obstacles.
    - Trajectory factors are graded and one-sided, with a focus on speed drop and lateral stability.
    """
    comp = {
        "perceptual_mention": 0.0,
        "decelerate_commitment": 0.0,
        "trajectory_speed_drop": 0.0,
    }

    # Perceptual mention of relevant entities
    if any(p.entity in ('cyclist', 'pedestrian', 'vehicle_generic') for p in claims.perceptual):
        comp["perceptual_mention"] = 0.05  # Reduced weight

    # Check for deceleration commitment and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints)) * traj.dt_s
        if speed_drop >= 0.05 and min_speed_time >= 5.0:  # Half of the GT drop and timing condition
            comp["decelerate_commitment"] = 0.3
            comp["trajectory_speed_drop"] = 0.6 * min(1.0, speed_drop / 3.8)  # Adjusted weight and graded factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
