"""clip 34ad49b8-ff38-4898-a914-37241266d140 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene 34ad49b8-ff38-4898-a914-37241266d140:
    - Deceleration in response to construction cones ahead.
    - Perceptual mention of construction-related entities.
    - Trajectory speed drop of at least 2.15 m/s (half of the positive case's 4.3 m/s drop).
    """

    # Initialize component scores
    perceptual_credit = 0.0
    deceleration_credit = 0.0

    # Check for perceptual mentions of construction-related entities
    if any(p.entity in ('construction_cones', 'barricades', 'work_zone', 'workers') for p in claims.perceptual):
        perceptual_credit = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration, requiring at least half the positive case's drop
        deceleration_credit = 0.7 * min(1.0, speed_drop / 4.3)

    return {
        "perceptual_mention": perceptual_credit,
        "deceleration_executed": deceleration_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
