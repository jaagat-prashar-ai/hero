"""clip b06169a9-2fb2-487e-a092-dfbf58d9bf5a - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene b06169a9-2fb2-487e-a092-dfbf58d9bf5a:
    Deceleration in response to a pedestrian ahead. Thresholds derived from
    GT: speed drop >= 1.0 m/s (half of 2.1 m/s GT drop) after pedestrian
    visibility begins (1.8s). Perceptual credit for mentioning pedestrian.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    deceleration_commitment = 0.0

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1  # Small additive weight for mention

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop after pedestrian becomes visible
        speed_window = window(traj.speed_mps, traj.dt_s, 1.8, traj.n_waypoints * traj.dt_s)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(speed_window)
        speed_drop = initial_speed - min_speed_after

        # Graded factor for deceleration
        deceleration_commitment = 0.6 * min(1.0, speed_drop / 2.0)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "deceleration_commitment": deceleration_commitment,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
