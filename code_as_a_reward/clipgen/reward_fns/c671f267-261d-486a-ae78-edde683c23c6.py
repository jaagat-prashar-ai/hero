"""clip c671f267-261d-486a-ae78-edde683c23c6 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring a rollout based on the decisive event of decelerating to maintain a safe distance from the lead vehicle. 
    - Perceptual mention of 'lead_vehicle' or 'vehicle_generic'.
    - Commitment to 'decelerate' with a graded speed drop.
    - Trajectory speed drop of at least 1.3 m/s, graded up to 2.6 m/s.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check for perceptual mention of lead vehicle
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_score = 0.1  # Small weight for mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        commitment_score = 0.3  # Larger weight for commitment

        # Calculate speed drop in the trajectory
        initial_speed = traj.initial_speed_mps
        min_speed_after = traj.min_speed_mps
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for speed drop
        trajectory_score = 0.6 * min(1.0, speed_drop / 2.6)  # Graded up to full drop

    # Return component scores
    return {
        "perceptual_mention": perceptual_score,
        "commitment_decelerate": commitment_score,
        "trajectory_speed_drop": trajectory_score,
    }

def reward(claims, traj):
    # Calculate total score and clamp between 0 and 1
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
