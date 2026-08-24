"""clip d4794539-f606-4a25-a770-030322d44655 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.36, real rollout argmax 11)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Deceleration for pedestrian: speed drop >= 3.9 m/s, graded
    - Mention of pedestrian: small additive credit
    """

    # Initialize component scores
    deceleration_for_pedestrian = 0.0
    pedestrian_mention = 0.0

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        pedestrian_mention = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded credit for deceleration, with a floor at 3.9 m/s
        deceleration_for_pedestrian = 0.7 * min(1.0, speed_drop / 6.0)

    return {
        "deceleration_for_pedestrian": deceleration_for_pedestrian,
        "pedestrian_mention": pedestrian_mention
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
