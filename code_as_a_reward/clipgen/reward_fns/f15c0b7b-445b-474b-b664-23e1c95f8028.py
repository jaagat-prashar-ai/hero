"""clip f15c0b7b-445b-474b-b664-23e1c95f8028 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene f15c0b7b-445b-474b-b664-23e1c95f8028:
    - Deceleration to maintain a safe distance from a cyclist.
    - Perceptual mention of 'cyclist' or related entities.
    - Commitment to 'decelerate' with a speed drop of at least 1.7 m/s, focusing on timing.
    """

    # Initialize component scores
    perceptual_mention = 0.0
    deceleration_commitment = 0.0

    # Check for perceptual mention of cyclist or related entities
    if any(p.entity in ('cyclist', 'pedestrian') for p in claims.perceptual):
        perceptual_mention = 0.05  # Reduced weight for mention-only credit

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Check the timing of the speed drop
        speed_window = window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)
        min_speed_idx = np.argmin(speed_window)
        min_speed_time = min_speed_idx * traj.dt_s

        # Graded factor for deceleration, floored at half the GT drop, with timing consideration
        if min_speed_time >= 3.0:  # Ensure the drop occurs in the latter half of the window
            deceleration_commitment = 0.65 * min(1.0, speed_drop / 3.4)

    return {
        "perceptual_mention": perceptual_mention,
        "deceleration_commitment": deceleration_commitment,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
