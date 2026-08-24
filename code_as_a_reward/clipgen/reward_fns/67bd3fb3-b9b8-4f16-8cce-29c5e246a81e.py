"""clip 67bd3fb3-b9b8-4f16-8cce-29c5e246a81e - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the scene's decisive events:
    1. Yielding for an emergency vehicle.
       - Commitment: 'decelerate' (yield)
       - Trajectory: Speed drop of at least 1.9 m/s
    """

    # Initialize component scores
    yield_executed = 0.0

    # Check for commitment to yield (decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        yield_executed = 0.7 * min(1.0, speed_drop / 1.9)

    return {
        "yield_executed": yield_executed,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
