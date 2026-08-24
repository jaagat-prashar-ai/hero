"""clip d26b27c4-5bf2-4060-b67c-9ce92025231f - attempt 3/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 10)"""
def components(claims, traj):
    """Components for scene with strong deceleration for a pedestrian.
    
    Decisive Event: Strong deceleration for pedestrian darting out to cross the road.
    - Perceptual: Mention of pedestrian or related entity.
    - Commitment: Deceleration (speed_profile='decelerate').
    - Trajectory: Speed drop of at least 1.15 m/s, graded with execution quality and timing.
    """
    perceptual_credit = 0.0
    commitment_credit = 0.0
    trajectory_credit = 0.0

    # Perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_credit = 0.1  # Small additive weight for mention

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop and check timing
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
        # Graded trajectory factor for deceleration with timing condition
        if min_speed_time <= 2.0:  # Ensure the deceleration happens early
            trajectory_credit = 0.5 * min(1.0, speed_drop / 2.3)
            # Combine with commitment presence
            commitment_credit = 0.4 if trajectory_credit > 0 else 0.0

    return {
        "perceptual_mention": perceptual_credit,
        "deceleration_commitment": commitment_credit,
        "trajectory_execution": trajectory_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
