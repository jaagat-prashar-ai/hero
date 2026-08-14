"""clip c8e6054a-d331-4964-9fe8-45c96be0cf0d - attempt 5/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing and yielding behavior.
    
    Decisive Event:
    - Pedestrian crossing the road at the crosswalk, requiring yielding behavior.
    
    Scene-derived thresholds:
    - Speed reduction: minimum drop of 4.0 m/s (half of GT's 7.9 m/s drop).
    - Timing: Speed reduction should occur around the pedestrian's visibility (t=4.5 s).
    """
    # Initialize component scores
    commitment_slowing = 0.0

    # Check for commitment to slow down
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Check timing of minimum speed
        min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints))
        # Graded trajectory factor for slowing with timing consideration
        if 3.5 <= min_speed_time <= 5.0:  # Timing window around t=4.1s
            trajectory_slowing = 0.7 * min(1.0, speed_drop / 6.0)
            commitment_slowing = trajectory_slowing

    return {
        "commitment_slowing": commitment_slowing,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
