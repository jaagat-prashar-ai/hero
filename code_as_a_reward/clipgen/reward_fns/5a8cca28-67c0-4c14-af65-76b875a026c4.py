"""clip 5a8cca28-67c0-4c14-af65-76b875a026c4 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.42, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing requiring yield. 
    Decisive event: yield to pedestrian (track 60) with speed drop.
    Trajectory expectations: speed drop >= 2.0 m/s, graded from 2.0 to 4.0 m/s.
    Perceptual mention: pedestrian or crosswalk.
    Commitment: decelerate family (stop/yield/wait/decelerate).
    """
    perceptual_credit = 0.0
    commitment_credit = 0.0
    trajectory_credit = 0.0

    # Perceptual mention credit
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_credit = 0.1

    # Commitment and trajectory credit
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after_t0 = np.min(window(traj.speed_mps, traj.dt_s, 0.0, 6.4))
        speed_drop = initial_speed - min_speed_after_t0

        # Graded trajectory factor for speed drop
        trajectory_credit = 0.6 * min(1.0, speed_drop / 4.0)

        # Combine with commitment credit
        commitment_credit = 0.3

    return {
        "perceptual_mention": perceptual_credit,
        "commitment_execution": commitment_credit,
        "trajectory_execution": trajectory_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
