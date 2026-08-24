"""clip ed630e63-261e-489d-9487-9797898c7d02 - attempt 5/5 - gate PASS (pos 0.99, max pert 0.59, real rollout argmax 9)"""
def components(claims, traj):
    """Components for navigating through a construction zone:
    - Commitment to accelerate through the zone, verified by trajectory.
    - Trajectory showing speed increase.
    """
    commitment_weight = 0.4
    trajectory_weight = 0.6

    # Commitment component: accelerate through the zone, verified by trajectory
    accelerate_commitment = any(
        c.speed_profile == 'accelerate' for c in claims.commitments
    )
    speed_increase = traj.final_speed_mps - traj.initial_speed_mps
    accelerate_execution = accelerate_commitment and speed_increase > 0.5  # Ensure speed increase

    # Trajectory component: speed increase
    speed_factor = min(1.0, speed_increase / 2.4)  # Graded factor with floor at ~half GT's increase

    components = {
        "accelerate_commitment": commitment_weight if accelerate_execution else 0.0,
        "trajectory_execution": trajectory_weight * speed_factor,
    }

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
