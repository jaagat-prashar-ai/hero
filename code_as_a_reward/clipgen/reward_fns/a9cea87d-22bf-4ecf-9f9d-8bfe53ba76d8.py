"""clip a9cea87d-22bf-4ecf-9f9d-8bfe53ba76d8 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Deceleration in response to the lead vehicle and traffic light.
    - Perceptual mention of 'lead_vehicle' or 'signal'.
    - Trajectory should show a speed drop of at least 2.0 m/s.
    """

    # Initialize component scores
    perceptual_score = 0.0
    deceleration_score = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('lead_vehicle', 'signal') for p in claims.perceptual):
        perceptual_score = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded deceleration score based on speed drop
        deceleration_score = 0.6 * min(1.0, speed_drop / 4.0)

    return {
        "perceptual_mention": perceptual_score,
        "deceleration_executed": deceleration_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
