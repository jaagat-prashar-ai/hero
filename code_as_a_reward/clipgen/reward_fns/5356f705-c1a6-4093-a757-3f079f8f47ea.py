"""clip 5356f705-c1a6-4093-a757-3f079f8f47ea - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of navigating
    through a construction zone by following temporary traffic cones. The scene
    requires maintaining a straight path with minimal lateral deviation and a
    slight speed increase. The trajectory factors are graded and one-sided,
    with commitment checks at the family level.
    """

    # Initialize component scores
    maintain_speed_or_accelerate = 0.0

    # Check for commitment to maintain speed or accelerate
    if any(c.speed_profile in ('maintain', 'accelerate') for c in claims.commitments):
        # Calculate the trajectory's speed increase
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        # Graded factor for maintaining or increasing speed
        maintain_speed_or_accelerate = 0.7 * min(1.0, speed_increase / 1.5)  # GT shows a 2.9 m/s increase

    # Return the component scores
    return {
        "maintain_speed_or_accelerate": maintain_speed_or_accelerate,
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of component scores
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
