"""clip 07906e8b-cb6e-414e-9ee3-29f1dd01489f - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of maintaining distance from pedestrians.
    Scene-derived thresholds:
    - Speed reduction: at least 0.75 m/s (half of the positive's 1.5 m/s drop)
    """

    # Initialize component scores
    slowing_commitment = 0.0

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = min(window(traj.speed_mps, traj.dt_s, 3.4, 6.4))  # After pedestrians are closest
        speed_drop = initial_speed - min_speed

        # Graded factor for slowing
        slowing_commitment = 0.7 * min(1.0, speed_drop / 0.75)  # Graded based on speed drop

    # Return component scores
    return {
        "slowing_commitment": slowing_commitment
    }

def reward(claims, traj):
    # Calculate the total score from components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
