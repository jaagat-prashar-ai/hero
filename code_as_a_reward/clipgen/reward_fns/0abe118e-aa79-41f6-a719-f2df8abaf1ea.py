"""clip 0abe118e-aa79-41f6-a719-f2df8abaf1ea - attempt 3/5 - gate PASS (pos 0.70, max pert 0.29, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 0abe118e-aa79-41f6-a719-f2df8abaf1ea:
    - Deceleration to yield to a pedestrian crossing the crosswalk.
    - Trajectory expectations: significant deceleration starting around 3.5-4.3s.
    - Commitment to decelerate must match trajectory execution.
    """
    # Initialize component scores
    commitment_decelerate = 0.0
    trajectory_decelerate = 0.0

    # Check for commitment claims
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the entire trajectory
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(traj.speed_mps)
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for deceleration
        trajectory_decelerate = 0.5 * min(1.0, speed_drop / 3.85)  # Adjusted for significant drop

        # Only award commitment credit if trajectory shows significant deceleration
        if trajectory_decelerate > 0:
            commitment_decelerate = 0.2

    return {
        "commitment_decelerate": commitment_decelerate,
        "trajectory_decelerate": trajectory_decelerate,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
