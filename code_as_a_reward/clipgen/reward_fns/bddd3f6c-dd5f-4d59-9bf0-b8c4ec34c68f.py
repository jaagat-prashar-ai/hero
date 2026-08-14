"""clip bddd3f6c-dd5f-4d59-9bf0-b8c4ec34c68f - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on:
    - Deceleration in response to pedestrian proximity.
    - Trajectory speed drop.
    - Perceptual mention of pedestrians.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    deceleration_executed = 0.0

    # Check for perceptual claim of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for deceleration commitment and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Ensure the minimum speed occurs later in the window
        min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))

        # Graded factor for speed drop, conditioned on timing
        if min_speed_time > 3.0:  # Ensure the minimum speed occurs later in the window
            deceleration_executed = 0.6 * min(1.0, speed_drop / 3.9)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "deceleration_executed": deceleration_executed,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
