"""clip 297759a3-520f-4c90-b921-8a770d0265e9 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.13, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene 297759a3-520f-4c90-b921-8a770d0265e9:
    - Deceleration in response to obstacles: expect 'decelerate' commitment
      and a speed drop of at least 2.85 m/s (half of the positive's 5.7 m/s).
    - Perceptual mentions: expect mention of 'pedestrian' or 'vehicle_generic'.
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
        deceleration_score = 0.9 * min(1.0, speed_drop / 5.7)

    return {
        "perceptual_mention": perceptual_score,
        "deceleration_executed": deceleration_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
