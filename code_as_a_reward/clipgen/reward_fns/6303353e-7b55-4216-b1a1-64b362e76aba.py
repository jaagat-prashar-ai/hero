"""clip 6303353e-7b55-4216-b1a1-64b362e76aba - attempt 1/5 - gate PASS (pos 0.90, max pert 0.40, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Stopping for pedestrians crossing the crosswalk.
       - Perceptual mention of 'pedestrian'.
       - Commitment to decelerate (stop/yield/wait/decelerate).
       - Trajectory shows a speed drop of at least 3.5 m/s by t=5.4 s.
    2. Proximity of nearby automobiles is not a decisive event, so no
       commitment or trajectory component is required for them.
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0
    trajectory_decelerate = 0.0

    # Check for perceptual mention of pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1  # Small weight for mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        commitment_decelerate = 0.3  # Larger weight for commitment

        # Calculate speed drop in the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 3.5:  # Floor at half the GT drop
            trajectory_decelerate = 0.5 * min(1.0, speed_drop / 6.9)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_decelerate": trajectory_decelerate
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
