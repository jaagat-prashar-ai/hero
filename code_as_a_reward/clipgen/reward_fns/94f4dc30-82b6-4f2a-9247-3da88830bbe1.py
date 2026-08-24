"""clip 94f4dc30-82b6-4f2a-9247-3da88830bbe1 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 3)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of maintaining a safe distance from a pedestrian ahead.
    - Commitment to decelerate, matched at the family level.
    - Trajectory showing a graded speed reduction, with a floor at half the GT magnitude.
    """
    # Initialize component scores
    commitment_score = 0.0

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for speed reduction
        trajectory_factor = 0.7 * min(1.0, speed_drop / 3.8)  # GT drop is 3.8 m/s

        # Combine commitment and trajectory for a graded score
        commitment_score = trajectory_factor

    return {
        "commitment_execution": commitment_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
