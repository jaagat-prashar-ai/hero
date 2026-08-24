"""clip 502cefc6-5c42-4146-bce7-41b7813b114e - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scene 502cefc6-5c42-4146-bce7-41b7813b114e:
    - Deceleration in anticipation of traffic barrels ahead.
    - Perceptual mention of traffic-related obstacles (e.g., barricades).
    - Graded speed reduction expectation with a floor at 0.5 m/s.
    """

    # Initialize component scores
    perceptual_mention = 0.0
    deceleration_commitment = 0.0

    # Check for perceptual mention of traffic-related obstacles
    if any(p.entity in ('barricades', 'construction_cones', 'work_zone') for p in claims.perceptual):
        perceptual_mention = 0.1  # Small weight for mention

    # Check for deceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop, with a floor at 0.5 m/s
        deceleration_commitment = 0.6 * min(1.0, speed_drop / 1.0)  # Expecting at least 0.5 m/s drop

    return {
        "perceptual_mention": perceptual_mention,
        "deceleration_commitment": deceleration_commitment
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
