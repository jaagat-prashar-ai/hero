"""clip 0d7556a6-264f-4b2b-a25b-367bcd1e22b2 - attempt 3/5 - gate PASS (pos 0.80, max pert 0.00, real rollout argmax 8)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive events:
    1. Deceleration to maintain a safe distance from the motorcyclist ahead.
       - Commitment: speed_profile='decelerate'
       - Trajectory: Speed drop of at least 0.8 m/s within the first 6.4 seconds.
    """
    # Initialize component scores
    comp = {
        "decelerate_commitment": 0.0,
        "speed_drop_execution": 0.0
    }

    # Commitment to decelerate with matching trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop within the first 6.4 seconds
        speed_window = window(traj.speed_mps, traj.dt_s, 0.0, 6.4)
        initial_speed = traj.initial_speed_mps
        min_speed = np.min(speed_window) if len(speed_window) > 0 else initial_speed
        speed_drop = initial_speed - min_speed

        # Graded factor for speed drop
        if speed_drop >= 0.8:
            comp["decelerate_commitment"] = 0.3
            comp["speed_drop_execution"] = 0.5 * min(1.0, speed_drop / 1.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
