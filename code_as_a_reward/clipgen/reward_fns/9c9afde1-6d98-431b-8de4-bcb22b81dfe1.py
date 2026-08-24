"""clip 9c9afde1-6d98-431b-8de4-bcb22b81dfe1 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Deceleration in response to the lead vehicle ahead.
    - Perceptual mention of a vehicle entity.
    - Trajectory speed drop of at least 5.5 m/s, graded.
    """

    # Initialize component scores
    perceptual_credit = 0.0
    deceleration_credit = 0.0

    # Check for perceptual mention of a vehicle entity
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_credit = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop over the trajectory
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded credit for deceleration based on speed drop
        # Adjusted to ensure it requires both a commitment claim and a trajectory execution
        deceleration_credit = 0.7 * min(1.0, speed_drop / 9.3)

    # Return the component scores
    return {
        "perceptual_mention": perceptual_credit,
        "deceleration_executed": deceleration_credit
    }

def reward(claims, traj):
    # Sum the component scores and clamp between 0.0 and 1.0
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
