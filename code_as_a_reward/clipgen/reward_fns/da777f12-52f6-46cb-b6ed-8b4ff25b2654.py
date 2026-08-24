"""clip da777f12-52f6-46cb-b6ed-8b4ff25b2654 - attempt 2/5 - gate PASS (pos 0.81, max pert 0.37, real rollout argmax 11)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Commitment to decelerate, with graded trajectory speed reduction.
    - Lateral offset maintenance conditioned on a deceleration commitment.
    """
    # Initialize component scores
    deceleration_commitment = 0.0
    lateral_maintenance = 0.0

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed reduction
        deceleration_commitment = 0.6 * min(1.0, speed_drop / 1.3)  # Floor at half of the positive's 2.6 m/s drop

        # Lateral offset maintenance conditioned on deceleration commitment
        max_lateral_offset = np.max(np.abs(window(traj.lateral_offset_m, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s)))
        if max_lateral_offset <= 0.38:  # GT's max |offset| is 0.38 m
            lateral_maintenance = 0.4 * (1.0 - (max_lateral_offset / 0.38))  # Graded factor

    return {
        "deceleration_commitment": deceleration_commitment,
        "lateral_maintenance": lateral_maintenance
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
