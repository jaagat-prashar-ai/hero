"""clip 9d8df2ed-c9ec-4291-b7f0-50e9bf2895b6 - attempt 3/5 - gate PASS (pos 0.81, max pert 0.01, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event:
    - Stop sign at the intersection.
    - Thresholds derived from the ground-truth trajectory: significant speed drop
      within the window, indicating preparation to stop.
    """

    # Initialize component scores
    commitment_decelerate = 0.0

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop within the window
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for deceleration
        # Adjusted to reflect the significant speed drop observed in the positive case
        trajectory_deceleration = 0.9 * min(1.0, speed_drop / 2.0)

        # Combine commitment and trajectory for deceleration
        commitment_decelerate = 0.9 * trajectory_deceleration

    return {
        "commitment_decelerate": commitment_decelerate,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
