"""clip 8b5e9f6f-2095-4919-8d66-1e13ba6b06c3 - attempt 4/5 - gate PASS (pos 1.00, max pert 0.29, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with yielding to an animal crossing the road.
    
    Decisive event: Yielding to an animal crossing the road.
    - Commitment family: decelerate (yield/stop/wait/decelerate)
    - Trajectory: Speed drop of at least 4.5 m/s within the first few seconds,
      heading change of at least -5.0 degrees.
    """
    slowing_claim = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    speed_series = np.array(traj.speed_mps)
    initial_speed = traj.initial_speed_mps
    min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, 6.4))
    speed_drop = initial_speed - min_speed_after

    slowing_execution = 0.7 * min(1.0, speed_drop / 4.5) if slowing_claim else 0.0

    heading_series = np.array(traj.heading_deg)
    initial_heading = heading_series[0]
    min_heading_after = np.min(window(heading_series, traj.dt_s, 0.0, 6.4))
    heading_change = initial_heading - min_heading_after

    heading_execution = 0.3 * min(1.0, abs(heading_change) / 5.0) if slowing_claim else 0.0

    return {
        "slowing_execution": slowing_execution,
        "heading_execution": heading_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
