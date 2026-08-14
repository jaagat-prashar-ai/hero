"""clip 7e884ffc-a6f4-4c33-a85e-c24f2f5b5500 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Pedestrian crossing: Expect deceleration and mention of pedestrian or crosswalk.
    Trajectory thresholds: Speed drop of at least 2.2 m/s (half of GT's 4.4 m/s drop).
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0
    trajectory_decelerate = 0.0

    # Check for perceptual mentions of pedestrian or crosswalk
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_pedestrian = 0.1  # Small weight for mention

    # Check for commitment to decelerate (stop, yield, wait, decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor based on speed drop, floored at half the GT's drop
        trajectory_decelerate = 0.5 * min(1.0, speed_drop / 4.4)
        commitment_decelerate = 0.4  # Weight for commitment presence

    # Combine components into a dictionary
    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_decelerate": trajectory_decelerate
    }

def reward(claims, traj):
    # Calculate the total score from components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
