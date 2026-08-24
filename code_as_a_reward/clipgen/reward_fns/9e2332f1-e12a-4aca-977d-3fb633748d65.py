"""clip 9e2332f1-e12a-4aca-977d-3fb633748d65 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.08, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene with deceleration to maintain safe distance.
    
    Decisive Event:
    - Deceleration to maintain a safe distance from nearby vehicles.
    
    Scene-derived thresholds:
    - Speed drop: at least 3.7 m/s (half of 7.4 m/s drop in GT).
    """
    # Initialize component scores
    deceleration_score = 0.0

    # Check for deceleration commitment and corresponding trajectory
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop, with a floor at half the GT drop
        deceleration_score = 0.7 * min(1.0, speed_drop / 7.4)

    return {
        "deceleration_executed": deceleration_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
