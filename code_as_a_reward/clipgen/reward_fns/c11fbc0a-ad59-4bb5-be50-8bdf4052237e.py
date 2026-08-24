"""clip c11fbc0a-ad59-4bb5-be50-8bdf4052237e - attempt 4/5 - gate PASS (pos 0.90, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Decelerate: Expect a speed drop of at least 4.0 m/s.
    """

    # Initialize component scores
    decelerate = 0.0

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps

        # Graded factor for deceleration
        decelerate = 0.9 * min(1.0, speed_drop / 8.0)

    return {
        "decelerate": decelerate
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
