"""clip 92ecb09a-b400-4086-a8d5-c595611f05fd - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 3)"""
def components(claims, traj):
    """
    Components for scene 92ecb09a-b400-4086-a8d5-c595611f05fd:
    - Decisive event: Presence of pedestrians on the right, requiring the ego vehicle to remain nearly stationary.
    - Commitment: Speed profile 'decelerate' (stop/yield/wait/decelerate).
    - Trajectory: Minimal speed change and lateral movement, reflecting nearly stationary behavior.
    - Trajectory thresholds: Lateral offset floor at 1.5 m, graded factor based on lateral movement.
    """
    # Check for a decelerate commitment
    decelerate_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Calculate the lateral offset
    lateral_offset = traj.final_lateral_offset_m

    # Graded factor for lateral offset
    lateral_offset_factor = 0.7 * min(1.0, lateral_offset / 3.0) if decelerate_commitment else 0.0

    # Components dictionary
    components = {
        "decelerate_commitment": lateral_offset_factor
    }

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
