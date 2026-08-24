"""clip b8befd8d-b5ff-4659-9182-89e4d5c4a109 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 4)"""
def components(claims, traj):
    """Components for the decisive event of decelerating due to proximity of the lead vehicle in a construction zone.
    - Perceptual: Mentions of 'lead_vehicle', 'work_zone', 'construction_cones'.
    - Commitment: Deceleration (speed_profile='decelerate') with graded speed drop.
    - Trajectory: Speed drop of at least 2.65 m/s, graded above this floor.
    """
    perceptual_credit = 0.0
    commitment_credit = 0.0
    speed_drop_credit = 0.0

    # Perceptual credit for mentioning relevant entities
    if any(p.entity in ('lead_vehicle', 'work_zone', 'construction_cones') for p in claims.perceptual):
        perceptual_credit = 0.1

    # Commitment credit for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded speed drop credit, requiring at least 2.65 m/s drop
        speed_drop_credit = 0.6 * min(1.0, speed_drop / 5.3)

        # Combine commitment and trajectory credit
        commitment_credit = speed_drop_credit

    return {
        "perceptual_mention": perceptual_credit,
        "deceleration_commitment": commitment_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
