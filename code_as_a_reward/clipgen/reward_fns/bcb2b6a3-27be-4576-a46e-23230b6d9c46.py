"""clip bcb2b6a3-27be-4576-a46e-23230b6d9c46 - attempt 5/5 - gate PASS (pos 0.91, max pert 0.10, real rollout argmax 4)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of maintaining distance from the lead vehicle.
    - Perceptual mention of 'lead_vehicle' or 'vehicle_generic'.
    - Commitment to 'maintain' speed profile, conditioned on trajectory execution.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_trajectory_score = 0.0

    # Perceptual mention of relevant entities
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_score = 0.1

    # Check for commitment to maintain speed and corresponding trajectory execution
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        # Calculate speed maintenance factor
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        # Ensure trajectory execution matches the commitment
        if final_speed >= initial_speed:
            trajectory_factor = 0.9 * min(1.0, (final_speed - initial_speed) / 1.5)
            commitment_trajectory_score = 0.9 * trajectory_factor

    return {
        "perceptual_mention": perceptual_score,
        "commitment_trajectory": commitment_trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
