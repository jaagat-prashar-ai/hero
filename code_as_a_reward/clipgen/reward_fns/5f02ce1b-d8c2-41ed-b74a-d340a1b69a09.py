"""clip 5f02ce1b-d8c2-41ed-b74a-d340a1b69a09 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.47, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with a pedestrian crossing at a crosswalk.
    Decisive event: stop for pedestrian. Thresholds: speed drop >= 0.65 m/s
    by t=3.5 s. Perceptual mention: pedestrian-related entity.
    Commitment: decelerate family (stop/yield/wait/decelerate)."""

    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0
    trajectory_slowing = 0.0

    # Perceptual check for pedestrian-related entities
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.05  # Reduced weight for mention-only credit

    # Commitment check for deceleration-related maneuvers
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory check for speed reduction
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, 3.5))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for slowing
        trajectory_slowing = 0.6 * min(1.0, speed_drop / 1.3)

        # Combine commitment and trajectory
        if trajectory_slowing > 0:
            commitment_decelerate = 0.35  # Increased weight to ensure conjunction

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_slowing": trajectory_slowing
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
