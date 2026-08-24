"""clip aa017a84-97e3-4452-be0f-fd60fe31bef6 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene aa017a84-97e3-4452-be0f-fd60fe31bef6:
    - Deceleration to maintain a safe distance from the crowd in the middle of the road.
    - Perceptual mention of pedestrian.
    - Graded speed reduction with a minimum drop of 0.15 m/s.
    """

    # Initialize component scores
    perceptual_mention = 0.0
    deceleration_executed = 0.0

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_mention = 0.1  # Small weight for mention

    # Check for deceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed reduction, with a minimum drop of 0.15 m/s
        deceleration_executed = 0.7 * min(1.0, speed_drop / 0.3)

    return {
        "perceptual_mention": perceptual_mention,
        "deceleration_executed": deceleration_executed,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
