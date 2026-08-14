"""clip e944196e-e743-4b9a-ab3e-5cd123420020 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.20, real rollout argmax 4)"""
def components(claims, traj):
    """
    Components for evaluating the rollout:
    - Perceptual mention of traffic signal.
    - Commitment to accelerate.
    - Execution of acceleration in trajectory, gated by commitment.
    Decisive events: Traffic light change and gentle acceleration.
    Scene-derived thresholds: Speed increase floor at 1.75 m/s (half of GT's 3.5 m/s).
    """

    # Initialize component scores
    perceptual_signal = 0.0
    commitment_accelerate = 0.0
    execution_accelerate = 0.0

    # Check for perceptual mention of traffic signal
    if any(p.entity in ('signal',) for p in claims.perceptual):
        perceptual_signal = 0.05  # Small weight for mention

    # Check for commitment to accelerate
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        commitment_accelerate = 0.15  # Weight for commitment presence

        # Calculate speed increase in trajectory, gated by commitment
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        if speed_increase > 0:
            # Graded factor for speed increase, floor at half of GT's increase
            execution_accelerate = 0.8 * min(1.0, speed_increase / 3.5)

    # Combine components
    return {
        "perceptual_signal": perceptual_signal,
        "commitment_accelerate": commitment_accelerate,
        "execution_accelerate": execution_accelerate
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
