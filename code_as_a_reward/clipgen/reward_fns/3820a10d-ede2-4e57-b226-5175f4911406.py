"""clip 3820a10d-ede2-4e57-b226-5175f4911406 - attempt 2/5 - gate PASS (pos 0.98, max pert 0.05, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Stop at the intersection: Expect a deceleration commitment and a speed drop of at least 1.65 m/s.
    2. Pedestrian interaction: Expect a deceleration commitment and a speed drop around the pedestrian's closest approach.
    Perceptual mentions are small additive components.
    """

    # Initialize component scores
    comp = {
        "perceptual_intersection": 0.0,
        "perceptual_pedestrian": 0.0,
        "commitment_stop_sign": 0.0,
        "commitment_pedestrian": 0.0,
    }

    # Perceptual mentions
    if any(p.entity in {'intersection', 'signal'} for p in claims.perceptual):
        comp["perceptual_intersection"] = 0.05

    if any(p.entity in {'pedestrian', 'crosswalk'} for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.05

    # Commitment and trajectory checks
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps

        # Stop sign related deceleration
        if speed_drop >= 1.65:
            comp["commitment_stop_sign"] = 0.5 * min(1.0, speed_drop / 3.3)

        # Pedestrian related deceleration
        # Assuming pedestrian interaction happens around t=2.9s
        pedestrian_window = window(traj.speed_mps, traj.dt_s, 2.3, 2.9)
        if len(pedestrian_window) > 0:
            min_speed_pedestrian = np.min(pedestrian_window)
            speed_drop_pedestrian = traj.initial_speed_mps - min_speed_pedestrian
            if speed_drop_pedestrian >= 1.65:
                comp["commitment_pedestrian"] = 0.5 * min(1.0, speed_drop_pedestrian / 3.3)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
