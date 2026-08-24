"""clip 7907a08c-26c0-4da6-b8c7-4859676bc518 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.01, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene with a decisive deceleration due to a deer crossing.
    - Deceleration commitment: speed_profile='decelerate', with a trajectory speed drop of at least 4.0 m/s.
    - Trajectory factors are graded and one-sided, with a generous floor.
    """

    # Initialize component scores
    deceleration_commitment = 0.0

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration, with a floor at 4.0 m/s
        deceleration_commitment = 0.7 * min(1.0, speed_drop / 8.0)

    return {
        "deceleration_commitment": deceleration_commitment,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
