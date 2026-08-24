"""clip a8b8012d-c65a-4556-a139-540ad44647ed - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 8)"""
def components(claims, traj):
    """Components for scene a8b8012d-c65a-4556-a139-540ad44647ed:
    - Strong deceleration for a red traffic light
    - Awareness of a cyclist ahead (mention-only)
    Trajectory thresholds:
    - Deceleration: speed drop >= 2.9 m/s, graded as 0.6 * min(1.0, speed_drop / 5.8)
    """

    # Initialize component scores
    deceleration_component = 0.0
    perceptual_credit = 0.0

    # Check for perceptual claims
    saw_signal = any(p.entity == 'signal' for p in claims.perceptual)
    saw_cyclist = any(p.entity == 'cyclist' for p in claims.perceptual)

    # Check for commitment claims
    intends_to_decelerate = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Calculate trajectory-based components
    if traj.n_waypoints > 0:
        # Deceleration component
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if intends_to_decelerate:
            deceleration_component = 0.6 * min(1.0, speed_drop / 5.8)

    # Perceptual mention-only credit
    perceptual_credit = 0.1 * saw_signal + 0.1 * saw_cyclist

    # Combine components
    return {
        "deceleration_component": deceleration_component,
        "perceptual_credit": perceptual_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
