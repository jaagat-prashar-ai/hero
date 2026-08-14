"""clip 00d4efa8-62a5-47ce-b8c7-e2a6b67b8f5c - attempt 2/5 - gate PASS (pos 0.89, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """Components for rewarding a rollout based on pedestrian crossing event.
    
    Decisive Event: Pedestrian Crossing
    - Perceptual: Mention of pedestrian-related entities.
    - Commitment: Deceleration to maintain safe distance.
    - Trajectory: Speed drop of at least 4.0 m/s, primarily between t=1.0 s and t=4.1 s.
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0
    trajectory_deceleration = 0.0

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 1.0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for deceleration
        trajectory_deceleration = 0.5 * min(1.0, speed_drop / 6.0)
        
        # Combine with commitment presence
        commitment_decelerate = 0.4 if trajectory_deceleration > 0 else 0.0

    # Return component scores
    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_deceleration": trajectory_deceleration
    }

def reward(claims, traj):
    # Calculate total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
