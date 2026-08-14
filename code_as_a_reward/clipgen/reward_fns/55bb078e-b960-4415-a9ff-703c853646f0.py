"""clip 55bb078e-b960-4415-a9ff-703c853646f0 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.30, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene with stopping/creeping behavior and proximity to nearby automobiles.
    
    Decisive events:
    1. Stopping/creeping behavior in response to nearby vehicles.
       - Commitment: speed_profile='decelerate'
       - Trajectory: speed drop of at least 1.2 m/s by t=3.1 s
    """
    # Initialize component scores
    comp = {
        "decelerate_commitment": 0.0,
        "speed_drop_execution": 0.0,
    }
    
    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.3  # Weight for commitment presence
    
    # Trajectory execution: speed drop
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    if any(c.speed_profile == 'decelerate' for c in claims.commitments) and speed_drop >= 1.2:  # Half of the GT speed drop
        comp["speed_drop_execution"] = 0.7 * min(1.0, speed_drop / 2.4)  # Graded factor
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
