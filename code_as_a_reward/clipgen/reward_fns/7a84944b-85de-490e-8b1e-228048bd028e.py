"""clip 7a84944b-85de-490e-8b1e-228048bd028e - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with automobiles on the right and slight speed adjustments.
    
    Decisive Events:
    1. Presence of automobiles on the right, requiring slight speed reduction.
    
    Scene-derived thresholds:
    - Speed drop: at least 0.8 m/s (half of measured 1.6 m/s drop), graded.
    """
    # Initialize component scores
    commitment_decelerate = 0.0

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop and check timing
        speed_series = np.array(traj.speed_mps)
        min_speed_idx = np.argmin(window(speed_series, traj.dt_s, 0, 6.4))
        min_speed_time = min_speed_idx * traj.dt_s
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop, conditioned on timing
        if 1.0 <= min_speed_time <= 3.0:  # Ensure the drop occurs early
            commitment_decelerate = 0.7 * min(1.0, speed_drop / 1.6)

    # Return component contributions
    return {
        "commitment_decelerate": commitment_decelerate
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
