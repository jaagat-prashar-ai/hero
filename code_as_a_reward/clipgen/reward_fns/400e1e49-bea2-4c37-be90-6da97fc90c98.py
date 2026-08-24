"""clip 400e1e49-bea2-4c37-be90-6da97fc90c98 - attempt 1/5 - gate PASS (pos 0.79, max pert 0.00, real rollout argmax 6)"""
def components(claims, traj):
    """Components for scene with pedestrians crossing and yielding action.
    
    Decisive Events:
    1. Pedestrians crossing the road at the crosswalk.
       - Perceptual mention: 'pedestrian', 'crosswalk'
       - Commitment: 'decelerate' family (stop/yield/wait/decelerate)
       - Trajectory: Speed drop of at least 3.75 m/s within the window.
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0
    trajectory_decelerate = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        trajectory_decelerate = 0.5 * min(1.0, speed_drop / 7.5)
        # Combine commitment and trajectory
        commitment_decelerate = 0.4 if trajectory_decelerate > 0 else 0.0

    # Return component scores
    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_decelerate": trajectory_decelerate
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
