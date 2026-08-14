"""clip 2fe669f8-677a-4432-9748-af24fcc01547 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.45, real rollout argmax 4)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of yielding to pedestrians at a crosswalk.
    - Perceptual: Mentions of pedestrians or crosswalk.
    - Commitment: Decelerate family commitment.
    - Trajectory: Speed reduction of at least 1.8 m/s, graded.
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    perceptual_crosswalk = 0.0
    commitment_decelerate = 0.0
    trajectory_slowing = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        perceptual_pedestrian = 0.05

    if any(p.entity in ('crosswalk',) for p in claims.perceptual):
        perceptual_crosswalk = 0.05

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for slowing
        trajectory_slowing = 0.5 * min(1.0, speed_drop / 3.6)
        # Ensure the timing of the minimum speed aligns with expected behavior
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
        if 3.0 <= min_speed_time <= 4.5:
            commitment_decelerate = 0.4

    # Return component scores
    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "perceptual_crosswalk": perceptual_crosswalk,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_slowing": trajectory_slowing
    }

def reward(claims, traj):
    # Calculate total score and clamp between 0 and 1
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
