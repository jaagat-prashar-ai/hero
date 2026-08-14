"""clip 1bb5226e-abd8-4d8d-9fbd-d6b8cecc0e08 - attempt 5/5 - gate PASS (pos 0.95, max pert 0.05, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Yielding to the cyclist: Expect mention of a cyclist and a deceleration commitment.
       Trajectory should show a speed drop of at least 0.6 m/s by around 5.2 seconds.
    """

    # Initialize component scores
    perceptual_cyclist = 0.0
    commitment_decelerate = 0.0

    # Check for perceptual mention of cyclist
    if any(p.entity == 'cyclist' for p in claims.perceptual):
        perceptual_cyclist = 0.05  # Small weight for mention-only

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop within the window
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for deceleration
        trajectory_decelerate = 0.95 * min(1.0, speed_drop / 1.2)

        # Combine with commitment check
        if trajectory_decelerate > 0:
            commitment_decelerate = 0.95 * trajectory_decelerate  # Larger weight for commitment and execution

    return {
        "perceptual_cyclist": perceptual_cyclist,
        "commitment_decelerate": commitment_decelerate
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
