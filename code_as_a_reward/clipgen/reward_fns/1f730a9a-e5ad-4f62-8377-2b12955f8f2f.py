"""clip 1f730a9a-e5ad-4f62-8377-2b12955f8f2f - attempt 4/5 - gate PASS (pos 0.91, max pert 0.05, real rollout argmax 3)"""
def components(claims, traj):
    """
    Components for scene 1f730a9a-e5ad-4f62-8377-2b12955f8f2f:
    - Decisive event: Slowing down for the lead vehicle.
    - Perceptual mention: Any of {'lead_vehicle', 'vehicle_generic'}.
    - Commitment family: 'decelerate' (stop/yield/wait/decelerate).
    - Trajectory: Speed drop of at least 2.4 m/s, graded, with timing consideration.
    """
    perceptual_credit = 0.0
    commitment_credit = 0.0

    # Perceptual mention of relevant entities
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_credit = 0.05  # Reduced weight for mention-only credit

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop and timing
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Find the time of minimum speed
        min_speed_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        min_speed_time = min_speed_idx * traj.dt_s

        # Graded trajectory factor for speed drop with timing consideration
        if min_speed_time <= 3.0:  # Ensure the drop happens early enough
            trajectory_credit = 0.95 * min(1.0, speed_drop / 2.4)  # Adjusted to ensure sufficient credit
            commitment_credit = trajectory_credit

    return {
        "perceptual_mention": perceptual_credit,
        "commitment_execution": commitment_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
