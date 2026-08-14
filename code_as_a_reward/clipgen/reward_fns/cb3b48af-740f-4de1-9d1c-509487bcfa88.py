"""clip cb3b48af-740f-4de1-9d1c-509487bcfa88 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring the rollout based on yielding to pedestrians.
    
    Decisive Event: Yield to pedestrians crossing the road.
    - Perceptual mention: 'pedestrian', 'crosswalk'
    - Commitment family: 'decelerate' (stop/yield/wait/decelerate)
    - Trajectory expectation: Speed reduction of at least 0.3 m/s during the window.
    """
    perceptual_credit = 0.1 if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual) else 0.0
    
    commitment_credit = 0.0
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for speed reduction
        trajectory_factor = 0.6 * min(1.0, speed_drop / 0.3)
        # Ensure the minimum speed occurs at the expected time
        min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        if 1.5 <= min_speed_time <= 3.0:
            commitment_credit = trajectory_factor
    
    return {
        "perceptual_mention": perceptual_credit,
        "yield_execution": commitment_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
