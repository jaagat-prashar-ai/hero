"""clip f81086b6-d4b3-47c5-a37a-fb71330b1105 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Yield to crossing traffic (trailer and automobile).
    - Perceptual mention of cross traffic or intersection.
    - Commitment to decelerate.
    - Trajectory showing a speed drop of at least 2.6 m/s.
    """
    # Initialize component scores
    perceptual_mention = 0.0
    deceleration_commitment = 0.0
    trajectory_deceleration = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('cross_traffic', 'intersection') for p in claims.perceptual):
        perceptual_mention = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for deceleration
        trajectory_deceleration = 0.6 * min(1.0, speed_drop / 5.2)

        # Combine commitment and trajectory
        deceleration_commitment = 0.3 if speed_drop >= 2.6 else 0.0

    # Return component scores
    return {
        "perceptual_mention": perceptual_mention,
        "deceleration_commitment": deceleration_commitment,
        "trajectory_deceleration": trajectory_deceleration
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
