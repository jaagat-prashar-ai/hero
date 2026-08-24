"""clip 14529dca-ea8e-4f4c-81a6-cbe365dfde7d - attempt 2/5 - gate PASS (pos 0.90, max pert 0.15, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene 14529dca-ea8e-4f4c-81a6-cbe365dfde7d:
    - Decelerate to yield to a pedestrian crossing the crosswalk.
    - Thresholds: speed drop >= 2.75 m/s (half of 5.5 m/s), occurring before t=5.3s.
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    perceptual_crosswalk = 0.0
    commitment_decelerate = 0.0

    # Check for perceptual claims
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        perceptual_pedestrian = 0.1  # Small additive weight for mentioning pedestrian

    if any(p.entity in ('crosswalk',) for p in claims.perceptual):
        perceptual_crosswalk = 0.05  # Small additive weight for mentioning crosswalk

    # Check for commitment claims
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 5.3))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for deceleration, with adjusted threshold and timing
        commitment_decelerate = 0.75 * min(1.0, speed_drop / 5.5)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "perceptual_crosswalk": perceptual_crosswalk,
        "commitment_decelerate": commitment_decelerate,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
