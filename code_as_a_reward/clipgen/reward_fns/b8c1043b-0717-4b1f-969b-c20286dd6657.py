"""clip b8c1043b-0717-4b1f-969b-c20286dd6657 - attempt 5/5 - gate PASS (pos 0.73, max pert 0.10, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scoring the rollout based on the decisive event of pedestrian crossing.
    
    Decisive Event: Pedestrian Crossing
    - Perceptual: Mention of 'pedestrian'
    - Commitment: Deceleration to maintain safe distance
    - Trajectory: Speed drop of at least 1.45 m/s by t=5.8 s
    
    Trajectory thresholds are one-sided and graded, with a focus on speed drop.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0

    # Check for perceptual mention of pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_score = 0.1  # Adjusted weight for mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for speed drop
        trajectory_factor = 0.9 * min(1.0, speed_drop / 2.9)  # Graded factor based on GT drop

        # Combine commitment and trajectory
        commitment_score = 0.7 * trajectory_factor  # Weight for commitment with trajectory

    # Return component contributions
    return {
        "perceptual_mention": perceptual_score,
        "commitment_execution": commitment_score,
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
