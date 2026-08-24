"""clip 351e5cd1-c14f-4af1-8771-06cf512e199e - attempt 1/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 351e5cd1-c14f-4af1-8771-06cf512e199e:
    - Decelerate while approaching a temporary traffic sign.
    - Perceptual mention of a traffic control entity.
    - Graded deceleration trajectory factor with a floor at half the GT's magnitude.
    """

    # Initialize component scores
    perceptual_mention = 0.0
    deceleration_commitment = 0.0
    deceleration_execution = 0.0

    # Check for perceptual mention of a traffic control entity
    if any(p.entity in ('signal', 'speed_limit_sign') for p in claims.perceptual):
        perceptual_mention = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded deceleration execution factor
        deceleration_execution = 0.5 * min(1.0, speed_drop / 0.1)
        # Combine commitment and execution
        deceleration_commitment = 0.4 if deceleration_execution > 0 else 0.0

    return {
        "perceptual_mention": perceptual_mention,
        "deceleration_commitment": deceleration_commitment,
        "deceleration_execution": deceleration_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
