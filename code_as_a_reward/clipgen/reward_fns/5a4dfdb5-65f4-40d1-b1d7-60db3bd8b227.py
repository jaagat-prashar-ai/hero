"""clip 5a4dfdb5-65f4-40d1-b1d7-60db3bd8b227 - attempt 2/5 - gate PASS (pos 0.90, max pert 0.20, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with cyclist yielding and nearby vehicles.
    
    Decisive Events:
    1. Yield to Cyclist: Expect mention of cyclist and deceleration.
       - Trajectory: Speed drop of at least 0.5 m/s, graded, with timing consideration.
    """
    # Initialize component scores
    comp = {
        "mention_cyclist": 0.0,
        "decelerate_for_cyclist": 0.0,
        "lateral_stability": 0.0
    }
    
    # Check for cyclist mention
    if any(p.entity == 'cyclist' for p in claims.perceptual):
        comp["mention_cyclist"] = 0.05  # Reduced weight for mention-only credit
    
    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop and timing
        initial_speed = traj.initial_speed_mps
        min_speed = min(window(traj.speed_mps, traj.dt_s, 0.1, 6.4))
        speed_drop = initial_speed - min_speed
        min_speed_time = 0.1 + np.argmin(window(traj.speed_mps, traj.dt_s, 0.1, 6.4)) * traj.dt_s
        # Graded factor for deceleration with timing consideration
        if 3.0 <= min_speed_time <= 5.0:  # Adjusted timing window
            comp["decelerate_for_cyclist"] = 0.7 * min(1.0, speed_drop / 1.0)
    
    # Check for lateral stability
    max_lateral_offset = max(abs(offset) for offset in window(traj.lateral_offset_m, traj.dt_s, 0.1, 6.4))
    if max_lateral_offset <= 0.1:
        comp["lateral_stability"] = 0.15  # Reduced weight, as it lacks commitment verification
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
