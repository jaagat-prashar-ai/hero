"""clip 656a0a19-e201-4222-8be7-33f87edb103f - attempt 2/5 - gate PASS (pos 0.80, max pert 0.12, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scoring the rollout based on stopping for a stop sign at an intersection.
    
    Decisive event: Stop for the stop sign.
    Scene-derived thresholds:
    - Perceptual mention of 'intersection' or related entities.
    - Commitment to 'decelerate' (stop/yield/wait/decelerate).
    - Trajectory speed drop of at least 2.65 m/s, with graded credit for greater drops.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0

    # Check for perceptual mention of relevant entities
    if any(p.entity in ('intersection', 'signal') for p in claims.perceptual):
        perceptual_score = 0.1  # Small additive weight for perceptual mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory score based on speed drop
        commitment_score = 0.7 * min(1.0, speed_drop / 5.3)  # GT drop is 5.3 m/s

    return {
        "perceptual_mention": perceptual_score,
        "commitment_execution": commitment_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
