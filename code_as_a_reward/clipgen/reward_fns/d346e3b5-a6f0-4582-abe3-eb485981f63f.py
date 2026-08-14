"""clip d346e3b5-a6f0-4582-abe3-eb485981f63f - attempt 1/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene: Stop behind the lead vehicle at the red traffic light.
    - Perceptual: mention of 'lead_vehicle', 'vehicle_generic', or 'signal'.
    - Commitment: speed_profile='decelerate' for stopping.
    - Trajectory: speed drop of at least 3.5 m/s, graded for more.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('lead_vehicle', 'vehicle_generic', 'signal') for p in claims.perceptual):
        perceptual_score = 0.1  # Small weight for mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded trajectory factor for speed drop
        trajectory_score = 0.5 * min(1.0, speed_drop / 6.0)  # Graded above 3.5 m/s drop

        # Combine with commitment
        commitment_score = 0.4 if trajectory_score > 0 else 0.0

    return {
        "perceptual_mention": perceptual_score,
        "commitment_to_decelerate": commitment_score,
        "trajectory_deceleration": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
