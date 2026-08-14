"""clip 04b4b3e7-216d-425c-8351-fb8045ecb609 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.16, real rollout argmax 6)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Deceleration and Stop: Expect a speed drop of at least 3.6 m/s.
    Perceptual mentions are expected for 'pedestrian' or 'vehicle_generic'.
    """
    # Initialize component scores
    deceleration_score = 0.0
    perceptual_score = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian', 'vehicle_generic') for p in claims.perceptual):
        perceptual_score = 0.1

    # Check for deceleration commitment and corresponding trajectory
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        deceleration_score = 0.9 * min(1.0, speed_drop / 7.2)

    return {
        "perceptual_mention": perceptual_score,
        "deceleration_executed": deceleration_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
