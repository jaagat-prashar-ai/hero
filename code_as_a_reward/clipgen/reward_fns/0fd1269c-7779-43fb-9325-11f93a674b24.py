"""clip 0fd1269c-7779-43fb-9325-11f93a674b24 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 9)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on:
    - Resuming speed after the intersection clears.
    - Speed increase trajectory factor.
    - Perceptual mention of relevant entities.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('intersection', 'vehicle_generic') for p in claims.perceptual):
        perceptual_score = 0.1  # Small weight for perceptual mention

    # Check for commitment to accelerate
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Calculate speed increase over the trajectory
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        # Graded factor for speed increase, floor at half the GT's increase
        commitment_score = 0.6 * min(1.0, speed_increase / 2.5)

    # Return the component scores
    return {
        "perceptual_mention": perceptual_score,
        "accelerate_commitment": commitment_score
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
