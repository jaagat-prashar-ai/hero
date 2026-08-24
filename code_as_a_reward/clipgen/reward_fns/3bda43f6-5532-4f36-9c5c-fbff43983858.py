"""clip 3bda43f6-5532-4f36-9c5c-fbff43983858 - attempt 1/5 - gate PASS (pos 0.74, max pert 0.30, real rollout argmax 10)"""
def components(claims, traj):
    """Decisive events: deceleration in response to an emergency vehicle,
    maintaining lateral position. Thresholds: speed drop >= 2.5 m/s,
    lateral offset within ±2.5 m."""
    
    # Initialize component scores
    perceptual_score = 0.0
    deceleration_score = 0.0
    lateral_position_score = 0.0
    
    # Check for perceptual mentions of relevant entities
    if any(p.entity in ('emergency_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_score = 0.1  # Small weight for perceptual mention
    
    # Check for deceleration commitment and corresponding trajectory
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration
        deceleration_score = 0.5 * min(1.0, speed_drop / 5.0)
    
    # Check for lateral position maintenance
    max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
    if max_lateral_offset <= 2.5:
        # Graded factor for maintaining lateral position
        lateral_position_score = 0.4 * min(1.0, (2.5 - max_lateral_offset) / 2.5)
    
    return {
        "perceptual_mention": perceptual_score,
        "deceleration_executed": deceleration_score,
        "lateral_position_maintained": lateral_position_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
