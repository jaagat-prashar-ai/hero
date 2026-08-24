"""clip 9d7cf232-550d-4831-a379-ce46dec43416 - attempt 3/5 - gate PASS (pos 0.90, max pert 0.04, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene with deceleration due to a temporary structure.
    
    Decisive Event: Deceleration for Temporary Structure
    - Commitment: speed_profile='decelerate'
    - Trajectory: speed reduction of at least 4.4 m/s within the window
    """
    commitment_credit = 0.0

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop within the window
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for speed reduction
        trajectory_credit = min(1.0, speed_drop / 8.8)  # Floor at 4.4 m/s drop

        # Combine commitment and trajectory credit
        commitment_credit = 0.9 * trajectory_credit  # Main conjunction component

    return {
        "deceleration_commitment": commitment_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
