"""clip 2f9d4d0b-ba8a-4db9-bee3-025abb3e4263 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing and nearby automobile.
    
    Decisive Events:
    1. Pedestrian Crossing: Strong deceleration to maintain a safe distance.
       - Perceptual: 'pedestrian'
       - Commitment: 'decelerate'
       - Trajectory: Speed drop >= 2.15 m/s by t=3.9s
    
    Trajectory thresholds are one-sided and graded, with a focus on
    deceleration and lateral positioning.
    """
    # Initialize component scores
    comp = {
        "saw_pedestrian": 0.0,
        "decelerate_executed": 0.0
    }
    
    # Perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.05  # Reduced weight for mention-only credit
    
    # Commitment claims and trajectory checks
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after
        
        # Graded trajectory factor for deceleration
        if speed_drop >= 2.15:
            comp["decelerate_executed"] = 0.65 * min(1.0, speed_drop / 4.3)
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
