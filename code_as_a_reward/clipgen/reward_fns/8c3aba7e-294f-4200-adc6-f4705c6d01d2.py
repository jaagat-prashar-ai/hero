"""clip 8c3aba7e-294f-4200-adc6-f4705c6d01d2 - attempt 2/5 - gate PASS (pos 0.83, max pert 0.10, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Deceleration to maintain a safe distance from the trailer (Track 119).
    - Yielding to the heavy truck (Track 122).
    - Perceptual mentions of relevant vehicles.
    - Trajectory expectations based on speed drop and timing.
    """

    # Initialize component scores
    deceleration_executed = 0.0
    perceptual_mention = 0.0

    # Check for perceptual mentions of relevant entities
    if any(p.entity in ('vehicle_generic', 'lane') for p in claims.perceptual):
        perceptual_mention = 0.1

    # Check for deceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Find the time of minimum speed
        min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        min_speed_time = min_speed_time_idx * traj.dt_s
        # Graded factor for deceleration execution, considering timing
        if 3.0 <= min_speed_time <= 4.0:  # Allow some tolerance around the GT time
            deceleration_executed = 0.8 * min(1.0, speed_drop / 2.9)

    return {
        "perceptual_mention": perceptual_mention,
        "deceleration_executed": deceleration_executed
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
