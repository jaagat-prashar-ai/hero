"""clip 8260e7c4-d797-4920-a98e-33e62158a7fd - attempt 2/5 - gate PASS (pos 0.90, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for evaluating the rollout:
    - Perception of traffic control devices (e.g., traffic light).
    - Commitment to decelerate (stop/yield/wait/decelerate).
    - Trajectory execution showing a speed drop consistent with stopping.
    - Lateral stability (staying in lane).
    """

    # Initialize component scores
    perception_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0
    lateral_stability_score = 0.0

    # Check for perception of traffic control devices
    if any(p.entity == 'signal' for p in claims.perceptual):
        perception_score = 0.1  # Small additive weight for perception

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        trajectory_score = 0.5 * min(1.0, speed_drop / 0.05)  # Floor at 0.05 m/s

        # Combine commitment and trajectory for deceleration
        commitment_score = 0.4 * trajectory_score  # Increased weight for commitment

    # Check for lateral stability (staying in lane)
    lateral_offset_change = abs(traj.final_lateral_offset_m - traj.lateral_offset_m[0])
    if lateral_offset_change < 0.5:  # Allow small lateral movement
        lateral_stability_score = 0.1  # Small weight for lateral stability

    return {
        "perception_traffic_control": perception_score,
        "commitment_decelerate": commitment_score,
        "trajectory_decelerate": trajectory_score,
        "lateral_stability": lateral_stability_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
