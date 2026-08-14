"""clip 42d0ecf9-f937-4121-bafa-742bd9bfa13c - attempt 1/5 - gate PASS (pos 0.95, max pert 0.25, real rollout argmax 1)"""
def components(claims, traj):
    """Decisive Event: Yield to the Pedestrian. Thresholds: speed drop >= 1.7 m/s, pedestrian mention."""
    
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    execution_score = 0.0

    # Check for perceptual mention of pedestrian
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        perceptual_score = 0.1  # Small weight for mention

    # Check for commitment to decelerate (yield/stop/wait/decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded execution score based on speed drop
        execution_score = 0.5 * min(1.0, speed_drop / 3.4)  # GT drop is 3.4 m/s

        # Only award commitment score if the trajectory shows deceleration
        if speed_drop >= 1.7:  # Half of GT drop
            commitment_score = 0.4  # Larger weight for commitment

    # Return component scores
    return {
        "perceptual_mention": perceptual_score,
        "commitment_and_execution": commitment_score + execution_score
    }

def reward(claims, traj):
    # Calculate total reward as the clamped sum of component scores
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
