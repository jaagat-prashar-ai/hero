"""clip ac26c8ae-f1c0-48c5-93e2-9c6fcbb08b37 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 9)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness in navigating through a construction zone.
    
    Decisive Event: Navigating through the construction zone with gentle acceleration.
    - Commitment: Accelerate family.
    - Trajectory: Speed increase of at least 0.1 m/s, graded above this floor.
    """
    # Check for acceleration commitment
    accelerate_commitment = any(c.speed_profile == 'accelerate' for c in claims.commitments)

    # Calculate the speed increase over the window
    speed_increase = traj.final_speed_mps - traj.initial_speed_mps
    graded_speeding = 0.7 * min(1.0, max(0.0, (speed_increase - 0.1) / 0.2)) if accelerate_commitment else 0.0

    return {
        "accelerate_executed": graded_speeding,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
