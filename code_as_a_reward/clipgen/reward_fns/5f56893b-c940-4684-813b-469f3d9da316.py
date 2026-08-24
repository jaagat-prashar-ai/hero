"""clip 5f56893b-c940-4684-813b-469f3d9da316 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 4)"""
def components(claims, traj):
    """
    Components for scene 5f56893b-c940-4684-813b-469f3d9da316:
    - Acceleration to pass the construction car.
    - Perceptual mention of a construction-related entity.
    - Speed increase of at least 4 m/s over the window.
    """

    # Initialize component scores
    perceptual_mention = 0.0
    acceleration_executed = 0.0

    # Check for perceptual mention of construction-related entities
    if any(p.entity in ('vehicle_generic', 'work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_mention = 0.1

    # Check for acceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Calculate the speed increase over the window
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        # Graded factor for acceleration execution
        acceleration_executed = 0.6 * min(1.0, speed_increase / 8.0)  # Floor at 4 m/s, graded up to 8 m/s

    return {
        "perceptual_mention": perceptual_mention,
        "acceleration_executed": acceleration_executed,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
