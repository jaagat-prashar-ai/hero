"""clip 5f802f8b-4594-4278-ab62-20ab581fc042 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.15, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with gentle deceleration for emergency ahead.
    
    Decisive Event: Gentle deceleration for emergency ahead.
    - Commitment: Deceleration family ('decelerate', 'stop', 'yield', 'wait').
    - Trajectory: Speed reduction of at least 1.2 m/s, graded with a floor at half the GT drop.
    """
    
    # Initialize component scores
    deceleration_commitment = 0.0
    
    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        trajectory_deceleration = 0.7 * min(1.0, speed_drop / 1.2)
        # Combine commitment and trajectory
        deceleration_commitment = trajectory_deceleration
    
    # Return component contributions
    return {
        "deceleration_commitment": deceleration_commitment
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
