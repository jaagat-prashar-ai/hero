"""clip 6f06996a-14f2-41c3-a4c5-380fa7abe099 - attempt 2/5 - gate PASS (pos 0.97, max pert 0.40, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scoring a rollout's faithfulness to the scene:
    - Deceleration in response to a pedestrian crossing.
    - Perceptual mention of pedestrian-related entities.
    - Trajectory deceleration graded from a 2.85 m/s floor.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    deceleration_commitment = 0.0
    trajectory_deceleration = 0.0

    # Check for perceptual mention of pedestrian-related entities
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        deceleration_commitment = 0.3

        # Calculate the speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop > 2.85:  # Half of the expert's 5.7 m/s drop
            trajectory_deceleration = 0.6 * min(1.0, speed_drop / 5.7)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "deceleration_commitment": deceleration_commitment,
        "trajectory_deceleration": trajectory_deceleration,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
