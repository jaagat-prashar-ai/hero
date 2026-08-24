"""clip 60fd2b97-b5bd-4f14-9e96-af9e330f95a3 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 4)"""
def components(claims, traj):
    """
    Components for scene 60fd2b97-b5bd-4f14-9e96-af9e330f95a3:
    - Deceleration for stop sign and pedestrian crossing.
    Trajectory thresholds are derived from the GT dossier:
    - Speed drop: 0.6 m/s (graded factor for deceleration commitment).
    """

    # Initialize component scores
    comp = {
        "decelerate_commitment": 0.0
    }

    # Deceleration commitment and trajectory check
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop within the window
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration, focusing on the timing of the minimum speed
        if np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s >= 5.0:
            comp["decelerate_commitment"] = 0.7 * min(1.0, speed_drop / 0.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
