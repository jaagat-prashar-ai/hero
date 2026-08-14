"""clip f6c6728e-ac1a-4a83-af9c-249b83f588bd - attempt 1/5 - gate PASS (pos 0.90, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of navigating through a construction zone with gentle acceleration.
    - Perceptual: Mentions of construction-related entities.
    - Commitment: Acceleration commitment with graded speed increase.
    - Trajectory: Speed increase reflecting gentle acceleration.
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Perceptual component: Check for mentions of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_score = 0.1

    # Commitment component: Check for acceleration commitment
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Trajectory component: Graded speed increase
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        speed_increase = final_speed - initial_speed
        trajectory_score = 0.6 * min(1.0, speed_increase / 2.6)  # Floor at half the GT's speed increase

        # Combine commitment and trajectory for acceleration
        commitment_score = 0.3 if trajectory_score > 0 else 0.0

    return {
        "perceptual_mention": perceptual_score,
        "commitment_accelerate": commitment_score,
        "trajectory_acceleration": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
