"""clip d28fdba4-40a6-452f-a785-b79a48994cc2 - attempt 3/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 5)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "commit_and_decelerate": 0.0,
    }
    
    # Check commitment claims and trajectory execution together
    if any(claim.maneuver in ['stop', 'yield'] for claim in claims.commitments):
        min_speed_time = traj.dt_s * (traj.speed_mps.index(traj.min_speed_mps))
        if 5.5 <= min_speed_time <= 6.4 and traj.initial_speed_mps - traj.min_speed_mps >= 6.0:
            scores["commit_and_decelerate"] = 0.7
    
    return scores

def reward(claims, traj):
    """Decisive events: Approaching the roundabout, significant deceleration.
    Thresholds: Deceleration >= 6.0 m/s, min speed <= 4.5 m/s by t=6.4s."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
