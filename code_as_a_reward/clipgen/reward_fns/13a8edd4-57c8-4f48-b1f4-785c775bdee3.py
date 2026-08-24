"""clip 13a8edd4-57c8-4f48-b1f4-785c775bdee3 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scene with a red traffic light and nearby vehicles.
    Decisive event: Decelerate for the red traffic light.
    Trajectory thresholds: speed drop >= 3.1 m/s (half of the positive case's 6.2 m/s).
    """

    # Initialize component scores
    commitment_decelerate = 0.0

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        commitment_decelerate = 0.7 * min(1.0, speed_drop / 3.1)

    return {
        "commitment_decelerate": commitment_decelerate
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
