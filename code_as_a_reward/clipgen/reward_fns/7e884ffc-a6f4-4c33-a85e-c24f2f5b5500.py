"""clip 7e884ffc-a6f4-4c33-a85e-c24f2f5b5500 - attempt 2/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events: yielding to a pedestrian crossing."""
    
    # Initialize component scores
    committed_and_executed_yield = 0.0
    
    # Check for commitment to yield and corresponding trajectory execution
    committed_to_yield = any(
        commitment.maneuver == 'yield' and commitment.speed_profile == 'decelerate'
        for commitment in claims.commitments
    )
    
    if committed_to_yield and traj.n_waypoints > 0:
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.final_speed_mps
        min_speed = min(traj.speed_mps)
        
        # Check if the trajectory shows a significant deceleration
        if speed_drop >= 4.2 and min_speed <= 0.5:
            committed_and_executed_yield = 0.7
    
    # Return component scores
    return {
        "committed_and_executed_yield": committed_and_executed_yield
    }

def reward(claims, traj):
    """Calculate the total reward based on component scores."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
