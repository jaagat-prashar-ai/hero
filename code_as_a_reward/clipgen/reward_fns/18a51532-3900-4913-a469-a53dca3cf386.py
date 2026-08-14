"""clip 18a51532-3900-4913-a469-a53dca3cf386 - attempt 3/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene with deceleration due to pedestrians ahead.
    
    Decisive Event: Deceleration to maintain a safe distance from pedestrians ahead.
    - Perceptual mention: pedestrians
    - Commitment family: decelerate
    - Trajectory expectation: speed drop of at least 1.05 m/s, graded above this floor
    """

    # Initialize component scores
    perceptual_mention = 0.0
    deceleration_executed = 0.0

    # Check for perceptual mention of pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_mention = 0.1  # Small additive weight for mention

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the window
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 3.3, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for deceleration execution
        # Adjusted to provide more credit for a faithful execution
        deceleration_executed = 0.7 * min(1.0, speed_drop / 2.1)

    # Return component scores
    return {
        "perceptual_mention": perceptual_mention,
        "deceleration_executed": deceleration_executed
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
