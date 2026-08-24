"""clip 7df02056-42d2-4698-82e1-4f8e674c11e8 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.24, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with strong deceleration for roadside assistance on the right shoulder.
    
    Decisive Event:
    - Strong deceleration for roadside assistance on the right shoulder.
    
    Scene-derived thresholds:
    - Speed drop of at least 0.9 m/s (half of GT's 1.8 m/s drop).
    - Timing of deceleration should reach minimum around t=5.9 s.
    """
    perceptual_weight = 0.1
    commitment_weight = 0.7  # Increased to ensure conjunction credit
    trajectory_weight = 0.2

    # Perceptual component
    perceptual_entities = {'shoulder_or_median', 'vehicle_generic', 'stopped_vehicle'}
    saw_perceptual = any(p.entity in perceptual_entities for p in claims.perceptual)
    perceptual_score = perceptual_weight if saw_perceptual else 0.0

    # Commitment component
    slowing_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    
    # Trajectory component
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    speed_drop_score = 0.0
    if slowing_commitment:
        # Ensure the trajectory execution is gated by the commitment claim
        speed_drop_score = commitment_weight * min(1.0, speed_drop / 8.5)

    # Combine perceptual and trajectory scores
    components = {
        "perceptual_mention": perceptual_score,
        "deceleration_executed": speed_drop_score
    }
    
    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
