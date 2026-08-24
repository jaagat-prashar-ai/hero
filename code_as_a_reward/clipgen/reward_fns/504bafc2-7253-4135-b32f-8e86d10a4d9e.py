"""clip 504bafc2-7253-4135-b32f-8e86d10a4d9e - attempt 3/5 - gate PASS (pos 0.99, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 504bafc2-7253-4135-b32f-8e86d10a4d9e:
    - Deceleration to maintain a safe distance from the lead vehicle.
    - Perceptual mention of lead vehicle.
    - Trajectory shows a speed drop of at least 0.65 m/s.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check for perceptual mention of the lead vehicle
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_score = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory score for deceleration, gated by commitment
        trajectory_score = 0.5 * min(1.0, speed_drop / 1.3)
        commitment_score = 0.4 if trajectory_score > 0 else 0.0

    # Combine components
    return {
        "perceptual_mention": perceptual_score,
        "deceleration_commitment": commitment_score,
        "trajectory_deceleration": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
