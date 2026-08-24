"""clip d9ef23e4-3a26-472a-a968-7acfa16bc888 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene.
    
    Decisive Events:
    1. Proximity of Automobiles on the Right and Left: Requires cautious deceleration.
       - Commitment family: 'decelerate'
       - Trajectory: Speed reduction of at least 1.65 m/s (half of 3.3 m/s drop), graded factor
    
    Scene-derived thresholds:
    - Speed drop threshold: 1.65 m/s (half of the measured 3.3 m/s drop)
    - Graded factor for speed drop: 0.7 * min(1.0, speed_drop / 3.3)
    """

    # Initialize component scores
    comp = {
        "decelerate_executed": 0.0,
    }

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop
        comp["decelerate_executed"] = 0.7 * min(1.0, speed_drop / 3.3)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
