"""clip 96651419-09fa-4dee-a9b7-6d458ce0fa34 - attempt 2/5 - gate PASS (pos 0.85, max pert 0.00, real rollout argmax 9)"""
def components(claims, traj):
    """Components for scoring the rollout based on resuming speed following the lead vehicle.
    
    Decisive Event: Resume speed following the lead vehicle while maintaining a safe distance.
    - Commitment: 'accelerate' family for resuming speed.
    - Trajectory: Speed increase from initial speed, with a graded factor.
    """
    commitment_weight = 0.7
    trajectory_weight = 0.3

    # Commitment component
    committed_to_accelerate = any(c.speed_profile == 'accelerate' for c in claims.commitments)
    
    # Trajectory component
    speed_increase = traj.final_speed_mps - traj.initial_speed_mps
    speed_factor = 0.5 * min(1.0, speed_increase / 2.0)  # Graded factor with a floor at 2.0 m/s increase

    commitment_score = commitment_weight if committed_to_accelerate and speed_increase > 0 else 0.0
    trajectory_score = trajectory_weight * speed_factor if committed_to_accelerate else 0.0

    return {
        "accelerate_commitment": commitment_score,
        "speed_increase_execution": trajectory_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
