"""clip 53307ce2-5835-49e2-83ef-085bfd17c5f6 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.13, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene 53307ce2-5835-49e2-83ef-085bfd17c5f6:
    - Deceleration for pedestrians: expect 'decelerate' commitment.
    - Trajectory: speed drop of at least 3.15 m/s, graded; focus on timing of minimum speed.
    """
    # Initialize component scores
    comp = {
        "decelerate_executed": 0.0
    }

    # Commitment claim checks
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory check for deceleration
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
        if min_speed_time <= 3.2:  # Focus on the timing of the deceleration
            comp["decelerate_executed"] = 0.7 * min(1.0, speed_drop / 6.3)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
