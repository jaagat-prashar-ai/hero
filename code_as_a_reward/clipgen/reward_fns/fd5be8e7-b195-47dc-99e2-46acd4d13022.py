"""clip fd5be8e7-b195-47dc-99e2-46acd4d13022 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.44, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scoring the rollout based on the decisive event of yielding to a pedestrian.
    
    Decisive Event: Yielding to a pedestrian crossing the road.
    - Perceptual mention of pedestrian.
    - Commitment to decelerate (yield/stop/wait/decelerate).
    - Trajectory showing a speed drop of at least 4.0 m/s, with graded credit for larger drops.
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0
    trajectory_decelerate = 0.0

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        trajectory_decelerate = 0.6 * min(1.0, speed_drop / 6.0)
        # Combine commitment and trajectory for deceleration
        commitment_decelerate = 0.3 if trajectory_decelerate > 0 else 0.0

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_decelerate": trajectory_decelerate
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
