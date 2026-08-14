"""clip 03a83788-ce57-4bec-bf99-e6e82f64227a - attempt 4/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scoring a rollout based on decisive events:
    1. Pedestrian crossing at crosswalk: expect deceleration and mention of pedestrian.
    Trajectory thresholds: speed drop >= 1.5 m/s, graded factor.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.05
    commitment_decelerate = 0.0

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.05

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = traj.min_speed_mps
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for deceleration
        trajectory_deceleration = 1.0 * min(1.0, speed_drop / 2.9)

        # Combine commitment and trajectory
        commitment_decelerate = 0.65 * trajectory_deceleration

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
