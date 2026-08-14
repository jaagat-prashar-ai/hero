"""clip 02fd6a8f-db9c-423e-beb6-37729e7e6af2 - attempt 2/5 - gate PASS (pos 0.71, max pert 0.30, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Stop for the stop sign held by the construction worker.
       - Perceptual mention: 'workers', 'pedestrian'
       - Commitment: speed_profile='decelerate'
       - Trajectory: Maintain near-zero speed, graded on speed drop.
    2. Maintain stationary/creeping position.
       - Trajectory: Minimal lateral movement.
    """
    # Initialize component scores
    perceptual_mention = 0.0
    stop_commitment = 0.0
    lateral_stability = 0.0

    # Check for perceptual mention of relevant entities
    if any(p.entity in ('workers', 'pedestrian') for p in claims.perceptual):
        perceptual_mention = 0.1

    # Check for commitment to decelerate (stop/yield/wait/decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the window
        initial_speed = traj.initial_speed_mps
        min_speed_after_t0 = min(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints))
        speed_drop = initial_speed - min_speed_after_t0
        # Graded factor for speed drop
        stop_commitment = 0.6 * min(1.0, speed_drop / 0.1)

    # Check for minimal lateral movement
    max_lateral_offset = max(abs(offset) for offset in window(traj.lateral_offset_m, traj.dt_s, 0, traj.n_waypoints))
    lateral_stability = 0.3 * min(1.0, (0.01 - max_lateral_offset) / 0.01)

    return {
        "perceptual_mention": perceptual_mention,
        "stop_commitment": stop_commitment,
        "lateral_stability": lateral_stability
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
