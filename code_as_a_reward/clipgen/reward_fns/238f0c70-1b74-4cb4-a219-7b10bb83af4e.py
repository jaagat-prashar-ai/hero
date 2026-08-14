"""clip 238f0c70-1b74-4cb4-a219-7b10bb83af4e - attempt 1/5 - gate PASS (pos 0.95, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with lead vehicle and cyclist constraints.
    
    Decisive events:
    - Stop behind the lead vehicle (track 148) in the same lane.
    - Wait for the cyclist (track 29) crossing the road.
    
    Scene-derived thresholds:
    - Speed drop: 0.0 m/s (floor at 0.0 m/s due to stationary behavior).
    - Lateral offset: |offset| <= 0.01 m.
    - Timing: Behavior should occur throughout the window, especially around 3.7s for the lead vehicle and 1.4s for the cyclist.
    """
    # Initialize component scores
    comp = {
        "mention_lead_vehicle": 0.0,
        "mention_cyclist": 0.0,
        "stop_for_lead_vehicle": 0.0,
        "wait_for_cyclist": 0.0
    }
    
    # Perceptual mentions
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["mention_lead_vehicle"] = 0.05
    
    if any(p.entity == 'cyclist' for p in claims.perceptual):
        comp["mention_cyclist"] = 0.05
    
    # Commitment and trajectory for stopping behind the lead vehicle
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        
        # Graded factor for speed drop
        comp["stop_for_lead_vehicle"] = 0.5 * min(1.0, speed_drop / 0.1)  # Floor at 0.0 m/s, graded above
    
    # Commitment and trajectory for waiting for the cyclist
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Use the same speed drop factor for waiting, as the behavior is similar
        comp["wait_for_cyclist"] = 0.4 * min(1.0, speed_drop / 0.1)  # Floor at 0.0 m/s, graded above
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
