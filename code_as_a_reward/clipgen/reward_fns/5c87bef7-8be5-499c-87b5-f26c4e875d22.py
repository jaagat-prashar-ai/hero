"""clip 5c87bef7-8be5-499c-87b5-f26c4e875d22 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.44, real rollout argmax 10)"""
def components(claims, traj):
    """
    Components for scoring a rollout's faithfulness to the scene:
    - Decisive Event: Pedestrian Presence Ahead
    - Perceptual: Mentions of 'pedestrian'
    - Commitment: Deceleration intent ('decelerate' family)
    - Trajectory: Speed drop of at least 1.25 m/s, graded above this floor
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0
    trajectory_decelerate = 0.0

    # Check for perceptual claims about pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1  # Small weight for mention

    # Check for commitment claims about deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        commitment_decelerate = 0.3  # Larger weight for commitment

        # Calculate the speed drop within the trajectory
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 3.2, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for deceleration
        trajectory_decelerate = 0.6 * min(1.0, speed_drop / 2.5)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_decelerate": trajectory_decelerate,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
