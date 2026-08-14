"""clip e7ff372b-14ae-4593-91d2-4dc12aee1ba0 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Yielding to the pedestrian crossing at the crosswalk.
       - Perceptual mention of 'pedestrian' or 'crosswalk'.
       - Commitment to 'decelerate' (stop/yield/wait/decelerate).
       - Trajectory should show a speed drop of at least 2.7 m/s by the end of the window.
    2. Stability in response to automobiles on the right.
       - Perceptual mention of 'vehicle_generic'.
       - No specific commitment required.
       - Trajectory should maintain lateral stability within ±0.22 m.
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    perceptual_vehicle = 0.0
    commitment_decelerate = 0.0
    trajectory_slowing = 0.0
    trajectory_stability = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        perceptual_vehicle = 0.05

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = traj.min_speed_mps
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for slowing
        trajectory_slowing = 0.5 * min(1.0, speed_drop / 5.4)
        commitment_decelerate = 0.35

    # Check for lateral stability
    lateral_offsets = np.array(traj.lateral_offset_m)
    max_lateral_deviation = np.max(np.abs(lateral_offsets))
    trajectory_stability = 0.05 * min(1.0, 0.22 / max_lateral_deviation)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "perceptual_vehicle": perceptual_vehicle,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_slowing": trajectory_slowing,
        "trajectory_stability": trajectory_stability
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
