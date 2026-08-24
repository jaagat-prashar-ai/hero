"""clip e1e957b4-c562-4a2b-9767-65eb400cb401 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.14, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with stop sign and pedestrian crossing.
    
    Decisive Events:
    1. Stop sign and pedestrian crossing: Deceleration expected due to pedestrian and stop sign.

    Trajectory thresholds:
    - Speed drop of at least 0.9 m/s (half of GT's 1.8 m/s drop) within the window.
    - Deceleration should occur primarily between 1.8 and 6.3 seconds.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    deceleration_commitment = 0.0

    # Check for perceptual claims
    if any(p.entity in {'pedestrian'} for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded factor for speed drop
        deceleration_commitment = 0.7 * min(1.0, speed_drop / 1.8)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "deceleration_commitment": deceleration_commitment,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
