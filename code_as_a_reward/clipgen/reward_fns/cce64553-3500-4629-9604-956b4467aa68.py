"""clip cce64553-3500-4629-9604-956b4467aa68 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 8)"""
def components(claims, traj):
    """Decisive Event: Strong Deceleration to Stop
    - Perceptual mention of intersection or construction-related entities.
    - Commitment to decelerate (speed_profile='decelerate').
    - Trajectory should show a speed drop of at least 2.9 m/s over the window.
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0

    # Check for relevant perceptual mentions
    if any(p.entity in ('intersection', 'barricades', 'construction_cones', 'work_zone') for p in claims.perceptual):
        perceptual_score = 0.05  # Small weight for perceptual mention

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the prediction window
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        trajectory_factor = min(1.0, speed_drop / 2.9)  # Half of GT drop is 2.9 m/s

        # Combine commitment and trajectory scores
        commitment_score = 0.65 * trajectory_factor  # Increased weight for commitment with trajectory execution

    # Return component contributions
    return {
        "perceptual_mention": perceptual_score,
        "deceleration_commitment": commitment_score
    }

def reward(claims, traj):
    # Calculate the total score from components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
