"""clip 525bbfa7-5bbc-4172-8633-d51d1c73345c - attempt 3/5 - gate PASS (pos 0.75, max pert 0.05, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with pedestrians crossing.
    Decisive events: deceleration to yield to pedestrians.
    Trajectory thresholds: speed drop >= 1.35 m/s, with graded factor for deceleration.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    deceleration_executed = 0.0

    # Check for perceptual claims
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        perceptual_pedestrian = 0.05  # Reduced weight for mention-only credit

    # Check for deceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed = min(traj.speed_mps)
        speed_drop = initial_speed - min_speed

        # Find the time of minimum speed
        min_speed_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        min_speed_time = min_speed_idx * traj.dt_s

        # Graded factor for deceleration execution, considering timing
        if 1.0 <= min_speed_time <= 3.0:  # Adjusted timing window
            deceleration_executed = 0.7 * min(1.0, speed_drop / 2.7)  # Adjusted weight and threshold

    # Return component scores
    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "deceleration_executed": deceleration_executed,
    }

def reward(claims, traj):
    # Sum the components and clamp the result between 0.0 and 1.0
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
