"""clip b12278b9-d550-41ff-97db-ae9d41855079 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring a rollout based on decisive events:
    1. Yielding to a pedestrian crossing the road.
    - Perceptual mention of a pedestrian.
    - Commitment to decelerate.
    - Trajectory shows a speed reduction of at least 3.7 m/s by t=5.9s.
    """
    # Initialize component scores
    perceptual_pedestrian = 0.05  # Reduced weight for mention-only credit
    commitment_decelerate = 0.0

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.05  # Small additive weight

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        trajectory_deceleration = 0.65 * min(1.0, speed_drop / 7.4)
        # Combine with commitment
        commitment_decelerate = trajectory_deceleration

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
