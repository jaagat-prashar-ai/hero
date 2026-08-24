"""clip 2b0a1f93-9f4d-4135-88a4-9386599b342f - attempt 5/5 - gate PASS (pos 0.95, max pert 0.30, real rollout argmax 6)"""
def components(claims, traj):
    """Components for scene with motorcycle cutting into the lane and automobile ahead.
    
    Decisive Events:
    1. Motorcycle cutting into the lane, requiring deceleration.
       - Commitment: speed_profile='decelerate'
       - Trajectory: Speed reduction of at least 0.15 m/s, graded factor
    
    2. Automobile ahead is background context, no specific commitment required.
    
    Trajectory thresholds derived from GT: speed drop of 0.3 m/s, graded from 0.15 m/s.
    """
    comp = {}

    # Commitment and trajectory component for deceleration
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    deceleration_claim = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    comp['decelerate_executed'] = 0.7 * deceleration_claim * min(1.0, speed_drop / 0.3)

    # Timing component for when the minimum speed occurs
    min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
    comp['timing_correct'] = 0.3 * (5.5 <= min_speed_time <= 6.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
