"""clip f40a7160-8026-4660-a30c-1e5a98b9a894 - attempt 1/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on maintaining speed while navigating around nearby vehicles.
    Decisive event: Maintaining speed with slight deceleration due to nearby vehicles, particularly track 177.
    Scene-derived thresholds: Speed drop of at least 0.2 m/s (half of GT's 0.4 m/s drop).
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check for perceptual claims related to vehicles
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        perceptual_score = 0.1  # Small weight for mention

    # Check for commitment claims related to maintaining speed
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for speed maintenance
        trajectory_score = 0.5 * min(1.0, speed_drop / 0.4)  # Floor at 0.2 m/s drop

        # Combine commitment and trajectory for maintaining speed
        commitment_score = 0.4 * (0.5 * min(1.0, speed_drop / 0.4))

    # Return component contributions
    return {
        "perceptual_mention": perceptual_score,
        "maintain_speed_commitment": commitment_score,
        "trajectory_execution": trajectory_score
    }

def reward(claims, traj):
    # Sum the components and clamp the result to [0, 1]
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
