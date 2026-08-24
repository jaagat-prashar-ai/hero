"""clip f0f1ff69-c546-4089-9680-725fba8b1552 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.29, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for evaluating the rollout based on decisive events:
    1. Proximity of automobiles on the left, requiring deceleration.
       - Commitment to 'decelerate'.
       - Trajectory should show a speed reduction of at least 2.2 m/s.
    """

    # Initialize component scores
    deceleration_executed = 0.0

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded factor for speed reduction, conditioned on the commitment
        deceleration_executed = 0.7 * min(1.0, speed_drop / 4.4)

    return {
        "deceleration_executed": deceleration_executed
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
