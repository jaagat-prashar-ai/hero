"""clip 3d598a9c-c657-4eac-9f6a-fbb31b531622 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scoring a rollout based on stopping for pedestrians at a stop sign.
    Decisive events: stopping for pedestrians, pedestrian presence.
    Trajectory thresholds: speed drop >= 1.15 m/s by t=3.4 s.
    """
    perceptual_pedestrian = any(p.entity == 'pedestrian' for p in claims.perceptual)
    
    commitment_decelerate = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    
    # Calculate speed drop within the window
    initial_speed = traj.initial_speed_mps
    min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
    speed_drop = initial_speed - min_speed_after
    
    # Graded trajectory factor for speed drop, considering timing
    min_speed_time_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
    min_speed_time = min_speed_time_idx * traj.dt_s
    speed_drop_factor = 0.6 * min(1.0, speed_drop / 2.3) if min_speed_time <= 3.4 else 0.0
    
    # Components
    components = {
        "mention_pedestrian": 0.1 if perceptual_pedestrian else 0.0,
        "decelerate_executed": speed_drop_factor if commitment_decelerate else 0.0,
    }
    
    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
