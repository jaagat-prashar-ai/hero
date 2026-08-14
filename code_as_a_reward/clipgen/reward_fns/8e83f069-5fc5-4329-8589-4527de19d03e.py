"""clip 8e83f069-5fc5-4329-8589-4527de19d03e - attempt 1/5 - gate PASS (pos 0.86, max pert 0.00, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scene with nearby automobiles requiring deceleration.
    - Decisive event: Deceleration due to nearby automobiles.
    - Perceptual entity family: vehicle_generic.
    - Commitment family: speed_profile='decelerate'.
    - Trajectory expectation: speed drop >= 2.65 m/s, graded factor.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Perceptual check: mention of nearby vehicles
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        perceptual_score = 0.1  # Small additive weight for mention

    # Commitment check: deceleration intent
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory check: graded speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        trajectory_score = 0.5 * min(1.0, speed_drop / 5.3)  # Graded factor based on speed drop

        # Combine commitment and trajectory
        commitment_score = 0.4 * (trajectory_score > 0)  # Only score if trajectory shows deceleration

    # Return component scores
    return {
        "perceptual_mention": perceptual_score,
        "deceleration_commitment": commitment_score,
        "trajectory_execution": trajectory_score
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
