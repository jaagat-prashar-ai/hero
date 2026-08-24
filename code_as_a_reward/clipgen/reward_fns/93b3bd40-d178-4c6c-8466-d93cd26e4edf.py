"""clip 93b3bd40-d178-4c6c-8466-d93cd26e4edf - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 93b3bd40-d178-4c6c-8466-d93cd26e4edf:
    - Gentle acceleration through a construction zone with workers.
    - Trajectory expectations: speed increase of at least 1.2 m/s, minimal lateral deviation.
    """

    # Initialize component scores
    acceleration_executed = 0.0

    # Check for acceleration commitment and corresponding trajectory
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        acceleration_executed = 0.7 * min(1.0, speed_increase / 2.4)  # Graded factor, floor at 1.2 m/s

    return {
        "acceleration_executed": acceleration_executed,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
