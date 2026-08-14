"""clip 8e06f83c-4de6-4cde-82ab-4a747645f80a - attempt 2/5 - gate PASS (pos 0.70, max pert 0.11, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with deceleration in response to pedestrians.
    
    Decisive Events:
    1. Deceleration in response to pedestrians crossing the road.
       - Perceptual mention: pedestrian, crosswalk
       - Commitment: speed_profile='decelerate'
       - Trajectory: Speed drop of at least 3.25 m/s by t=4.6 s
    """
    comp = {
        "perceptual_pedestrian": 0.0,
        "decelerate_execution": 0.0,
    }
    
    # Perceptual mention of pedestrians
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1  # Small additive weight for mention

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after
        
        # Graded factor for speed drop
        comp["decelerate_execution"] = 0.6 * min(1.0, speed_drop / 6.5)  # Graded, with floor at half GT drop

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
