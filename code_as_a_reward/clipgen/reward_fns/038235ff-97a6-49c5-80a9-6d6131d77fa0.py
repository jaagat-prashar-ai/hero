"""clip 038235ff-97a6-49c5-80a9-6d6131d77fa0 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.07, real rollout argmax 0)"""
def components(claims, traj):
    """Components for navigating through a construction zone:
    - Commitment to maintain or decelerate speed.
    - Trajectory execution showing speed reduction.
    """
    # Commitment to maintain or decelerate speed
    commitment_maintain_or_decelerate = any(
        c.speed_profile in ('maintain', 'decelerate') for c in claims.commitments
    )

    # Trajectory execution showing speed reduction
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    speed_reduction_factor = 0.7 * min(1.0, speed_drop / 2.5) if commitment_maintain_or_decelerate else 0.0

    return {
        "speed_reduction": speed_reduction_factor
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
