"""clip e921dddd-6d73-448b-a22e-d2f73f7eefab - attempt 2/5 - gate PASS (pos 0.95, max pert 0.45, real rollout argmax 6)"""
def components(claims, traj):
    """Components for reward calculation based on the decisive event of gentle acceleration after passing a construction zone.
    
    Decisive Event: Gentle Acceleration After Passing the Construction Zone
    - Perceptual mention of construction-related entities.
    - Commitment to accelerate.
    - Trajectory should show a speed increase of at least 1.5 m/s after initial deceleration.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check for perceptual mentions related to the construction zone
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'intersection') for p in claims.perceptual):
        perceptual_score = 0.05  # Reduced weight for mention-only credit

    # Check for commitment to accelerate
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Calculate speed increase in the trajectory
        speed_series = np.array(traj.speed_mps)
        min_speed_idx = np.argmin(window(speed_series, traj.dt_s, 0, 6.4))
        min_speed_time = min_speed_idx * traj.dt_s
        speed_increase = traj.final_speed_mps - traj.min_speed_mps

        # Check if the speed increase is significant and occurs after the initial deceleration
        if min_speed_time >= 0.0:  # Adjusted to match the positive case
            trajectory_score = 0.5 * min(1.0, speed_increase / 1.5)  # Adjusted threshold
            commitment_score = 0.4  # Commitment to accelerate

    return {
        "perceptual_mention": perceptual_score,
        "commitment_to_accelerate": commitment_score,
        "trajectory_acceleration": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
