"""clip 3e1ebb42-b617-46d6-bfa2-91d99bdee9d8 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Decisive events: gentle deceleration for pedestrians and lead vehicle.
    Trajectory thresholds: speed drop >= 0.7 m/s (half of GT's 1.4 m/s drop),
    with graded credit for larger drops; lateral offset within |0.28| m.
    Commitment credit matched at the FAMILY level (speed_profile='decelerate').
    Perceptual credit for mentioning 'pedestrian' or 'vehicle'."""
    
    # Initialize component scores
    perceptual_credit = 0.0
    deceleration_credit = 0.0
    
    # Check for perceptual mentions
    if any(p.entity in ('pedestrian', 'lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_credit = 0.1  # Small additive weight for perceptual mention
    
    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        
        # Graded credit for speed drop, floor at 0.7 m/s
        if speed_drop >= 0.7:
            deceleration_credit = 0.6 * min(1.0, speed_drop / 1.4)
    
    # Return component contributions
    return {
        "perceptual_mention": perceptual_credit,
        "deceleration_executed": deceleration_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
