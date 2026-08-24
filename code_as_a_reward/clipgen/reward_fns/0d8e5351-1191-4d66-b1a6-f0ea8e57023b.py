"""clip 0d8e5351-1191-4d66-b1a6-f0ea8e57023b - attempt 4/5 - gate PASS (pos 0.99, max pert 0.35, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Deceleration in response to a pedestrian crossing.
    - Perceptual mention of pedestrian-related entities.
    - Trajectory deceleration factor based on speed drop.
    Scene-derived thresholds:
    - Speed drop: at least 1.15 m/s (half of GT's 2.3 m/s drop).
    - Timing: Deceleration should occur around t=3.7 s.
    """

    # Initialize component scores
    perceptual_mention = 0.0
    deceleration_commitment = 0.0
    trajectory_deceleration = 0.0

    # Check for perceptual mention of pedestrian-related entities
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_mention = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory deceleration factor
        trajectory_deceleration = 0.4 * min(1.0, speed_drop / 2.3)
        # Check the timing of the minimum speed
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        min_speed_time = min_speed_time_idx * traj.dt_s
        # Combine with commitment presence and timing condition
        if trajectory_deceleration > 0 and 3.0 <= min_speed_time <= 4.5:
            deceleration_commitment = 0.5

    # Return component contributions
    return {
        "perceptual_mention": perceptual_mention,
        "deceleration_commitment": deceleration_commitment,
        "trajectory_deceleration": trajectory_deceleration
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
