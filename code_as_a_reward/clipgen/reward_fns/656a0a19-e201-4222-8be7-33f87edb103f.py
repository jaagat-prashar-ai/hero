"""clip 656a0a19-e201-4222-8be7-33f87edb103f - attempt 3/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "committed_to_stop_and_executed": 0.0,
    }
    
    # Check for commitment to stop and execution of stop
    if any(claim.maneuver == "stop" and claim.speed_profile == "decelerate" for claim in claims.commitments):
        # Check if the trajectory shows a significant speed drop at the correct time
        speed_window = window(traj.speed_mps, traj.dt_s, 6.0, 6.5)
        if len(speed_window) > 0 and min(speed_window) <= 0.5:
            scores["committed_to_stop_and_executed"] = 0.7
    
    return scores

def reward(claims, traj):
    """Reward function for stopping at a stop sign. Key checks:
    - Committing to stop and executing a stop (commitment claim + trajectory)
    """
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
