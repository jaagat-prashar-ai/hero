"""clip d3abc439-ac6c-48c5-b8be-d59dd70decc1 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring the rollout based on stopping/creeping at a traffic light scene.
    
    Decisive Event: Stopping/Creeping at the Traffic Light
    - Perceptual: Mention of 'lead_vehicle' or 'signal' is reasonable.
    - Commitment: A 'decelerate' speed profile is expected.
    - Trajectory: Maintain a low speed, with a final speed not exceeding 0.5 m/s.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Perceptual check: Mention of lead_vehicle or signal
    if any(p.entity in ('lead_vehicle', 'signal') for p in claims.perceptual):
        perceptual_score = 0.1  # Small weight for perceptual mention

    # Commitment check: Decelerate family
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory check: Maintain low speed
        final_speed = traj.final_speed_mps
        trajectory_score = 0.5 * min(1.0, (0.5 - final_speed) / 0.5)  # Graded factor for low speed
        commitment_score = 0.4  # Weight for commitment presence

    # Combine components into a dictionary
    return {
        "perceptual_mention": perceptual_score,
        "commitment_executed": commitment_score,
        "trajectory_executed": trajectory_score
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
