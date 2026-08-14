"""clip 5cdabd0b-9164-489a-b690-e384880d17cb - attempt 1/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with deceleration due to pedestrians.
    
    Decisive Event: Deceleration to maintain a safe distance from pedestrians.
    - Perceptual mention of 'pedestrian' expected.
    - Commitment to 'decelerate' expected.
    - Trajectory should show a speed drop of at least 2.0 m/s, graded.
    """
    # Initialize component scores
    perceptual_mention = 0.0
    deceleration_commitment = 0.0
    speed_drop_execution = 0.0

    # Check for perceptual mention of pedestrians
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        perceptual_mention = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed

        # Graded factor for speed drop execution
        speed_drop_execution = 0.5 * min(1.0, speed_drop / 4.0)

        # Combine commitment and execution
        deceleration_commitment = 0.4 if speed_drop_execution > 0 else 0.0

    # Return component scores
    return {
        "perceptual_mention": perceptual_mention,
        "deceleration_commitment": deceleration_commitment,
        "speed_drop_execution": speed_drop_execution
    }

def reward(claims, traj):
    # Calculate total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
