"""clip 2886f3db-0f25-478d-a30c-b9a5e9524485 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.00, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene 2886f3db-0f25-478d-a30c-b9a5e9524485:
    - Maintain safe distance from right-side automobiles by decelerating.
    - Deceleration threshold: speed drop >= 1.1 m/s by t=6.3 s.
    """
    deceleration_weight = 1.0

    # Commitment component: deceleration
    committed_to_decelerate = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    deceleration_score = 0.0
    if committed_to_decelerate:
        deceleration_score = deceleration_weight * min(1.0, speed_drop / 2.2)

    return {
        "deceleration_executed": deceleration_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
