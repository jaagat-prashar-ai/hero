"""clip 8cc3e867-3e28-4a63-a42f-2bd149e5964c - attempt 4/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with a stop for a red traffic light.
    - Decisive event: Stop for the red traffic light.
    - Perceptual mention: signal.
    - Commitment: decelerate family (stop/yield/wait/decelerate).
    - Trajectory: stop event should be true.
    """
    # Initialize component scores
    perceptual_mention = 0.0
    commitment_execution = 0.0

    # Check for perceptual mention of traffic signal
    if any(p.entity == 'signal' for p in claims.perceptual):
        perceptual_mention = 0.1  # Small additive weight for mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Check for stop event in the trajectory
        if traj.stop_event:
            commitment_execution = 0.7  # Full credit for executing stop

    return {
        "perceptual_mention": perceptual_mention,
        "commitment_execution": commitment_execution
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
