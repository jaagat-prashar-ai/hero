"""clip 3749d5ee-6aeb-4203-b223-f08e9d4f0bc2 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrians and vehicles:
    - Maintain speed while waiting for pedestrians to cross.
    - No significant lateral or speed change in response to vehicles.
    - Trajectory thresholds based on expert trajectory: speed increase expected.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    maintain_speed = 0.0

    # Check for perceptual claims
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        perceptual_pedestrian = 0.05  # Reduced weight for mention-only credit

    # Check for commitment claims and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed increase over the trajectory
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        # Graded factor for maintaining or slightly increasing speed
        maintain_speed = 0.65 * min(1.0, speed_increase / 3.0)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "maintain_speed": maintain_speed,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
