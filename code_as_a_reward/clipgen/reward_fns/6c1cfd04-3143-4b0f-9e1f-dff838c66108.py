"""clip 6c1cfd04-3143-4b0f-9e1f-dff838c66108 - attempt 5/5 - gate PASS (pos 0.71, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scene with a pedestrian crossing and nearby automobiles.
    Decisive event: yield to pedestrian, with a trajectory showing significant deceleration.
    Trajectory thresholds: speed drop >= 1.8 m/s for deceleration credit.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0

    # Check for perceptual claim of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.05  # Small weight for mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop and timing
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after_idx = np.argmin(window(speed_series, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        min_speed_after_time = min_speed_after_idx * traj.dt_s
        min_speed_after = speed_series[min_speed_after_idx]
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for deceleration with timing condition
        if min_speed_after_time <= 3.0:  # Ensure the drop happens early enough
            trajectory_deceleration = 0.95 * min(1.0, speed_drop / 1.8)  # Floor at 1.8 m/s drop
            commitment_decelerate = 0.70 * trajectory_deceleration  # Increased weight for conjunction

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
