"""clip 223bb101-ce1e-4fc3-ad3b-2297647d63f9 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with deceleration due to nearby vehicles.
    
    Decisive Events:
    1. Deceleration due to Track 26 (Automobile on the Right)
    2. Deceleration due to Track 15 (Automobile on the Right)
    
    Scene-derived thresholds:
    - Speed reduction of at least 1.45 m/s (half of GT's 2.9 m/s drop)
    - Timing around t=6.3 s for minimum speed
    """
    # Initialize component scores
    decelerate_commitment = 0.0

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed reduction
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded speed reduction factor
        speed_reduction = 0.7 * min(1.0, speed_drop / 1.45)  # Graded above 1.45 m/s drop

        # Combine commitment and trajectory execution
        decelerate_commitment = speed_reduction  # Weight for commitment + execution

    # Return component scores
    return {
        "decelerate_commitment": decelerate_commitment
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
