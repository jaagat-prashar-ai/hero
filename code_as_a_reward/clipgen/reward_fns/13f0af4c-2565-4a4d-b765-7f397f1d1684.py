"""clip 13f0af4c-2565-4a4d-b765-7f397f1d1684 - attempt 3/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring rollouts in a scene with pedestrians requiring deceleration.
    
    Decisive Events:
    1. Pedestrian Crossing (Track 18): Requires deceleration due to proximity.
    2. Pedestrian Crossing (Track 47): Also requires deceleration due to proximity.
    
    Scene-Derived Thresholds:
    - Speed drop of at least 0.7 m/s (half of GT's 1.4 m/s drop) after t=3.0 seconds.
    - Perceptual mention of 'pedestrian'.
    """
    # Initialize component scores
    scores = {
        "perceptual_mention": 0.0,
        "deceleration_executed": 0.0,
    }

    # Check for perceptual mentions of pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        scores["perceptual_mention"] = 0.1

    # Check for deceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after_t3 = np.min(window(traj.speed_mps, traj.dt_s, 3.0, 6.4))
        speed_drop = initial_speed - min_speed_after_t3
        
        # Graded score for deceleration
        scores["deceleration_executed"] = 0.7 * min(1.0, speed_drop / 1.4)

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
