"""clip 423bd3a2-2ea7-48d6-947e-d14318b1bddf - attempt 2/5 - gate PASS (pos 0.74, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene 423bd3a2-2ea7-48d6-947e-d14318b1bddf:
    - Decisive Event 1: Stopping for the red traffic light
      - Perceptual: signal
      - Commitment: speed_profile='decelerate'
      - Trajectory: Speed drop >= 1.85 m/s by t=5.1 s
    - Decisive Event 2: Presence of nearby vehicles (contextual)
      - Perceptual: vehicle_generic
      - No specific commitment required
    """
    # Initialize component scores
    comp = {
        "saw_signal": 0.0,
        "stop_executed": 0.0
    }

    # Check for perceptual claims
    if any(p.entity == 'signal' for p in claims.perceptual):
        comp["saw_signal"] = 0.1

    # Check for commitment claims and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for stopping
        comp["stop_executed"] = 0.7 * min(1.0, speed_drop / 3.7)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
