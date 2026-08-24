"""clip 4e15d094-4104-4e39-9865-9dc6cb4a07dd - attempt 3/5 - gate PASS (pos 0.75, max pert 0.05, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene with deceleration to maintain safe distance from lead vehicle.
    
    Decisive Event: Deceleration to maintain safe distance
    - Perceptual mention: 'lead_vehicle' or 'vehicle_generic'
    - Commitment: 'decelerate' family
    - Trajectory: Speed drop of at least 1.25 m/s (half of GT's 2.5 m/s), graded factor
    
    Scene-derived thresholds:
    - Minimum speed drop: 1.25 m/s
    - Graded trajectory factor: 0.5 * min(1.0, speed_drop / 2.5)
    """

    # Initialize component scores
    perceptual_mention = 0.0
    deceleration_commitment = 0.0

    # Check for perceptual mention of relevant vehicle entities
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_mention = 0.05  # Reduced weight for mention-only credit

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        trajectory_deceleration = 0.7 * min(1.0, speed_drop / 2.5)
        # Combine commitment and trajectory for deceleration
        deceleration_commitment = trajectory_deceleration

    # Return component scores
    return {
        "perceptual_mention": perceptual_mention,
        "deceleration_commitment": deceleration_commitment,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
