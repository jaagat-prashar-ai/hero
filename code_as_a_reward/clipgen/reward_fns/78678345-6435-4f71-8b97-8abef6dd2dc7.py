"""clip 78678345-6435-4f71-8b97-8abef6dd2dc7 - attempt 2/5 - gate PASS (pos 0.89, max pert 0.31, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Deceleration to stop in response to pedestrian presence.
       - Perceptual mention: 'pedestrian'
       - Commitment: speed_profile='decelerate'
       - Trajectory: speed drop of at least 1.1 m/s, graded
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0
    trajectory_decelerate = 0.0

    # Check for perceptual mention of pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        trajectory_decelerate = 0.6 * min(1.0, speed_drop / 2.2)
        # Combine with commitment presence
        commitment_decelerate = 0.3 if trajectory_decelerate > 0 else 0.0

    # Return the component scores
    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_decelerate": trajectory_decelerate
    }

def reward(claims, traj):
    # Sum the components and clamp the result between 0.0 and 1.0
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
