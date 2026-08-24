"""clip 8502724a-89a4-4c12-babd-ddd7301eacc9 - attempt 1/5 - gate PASS (pos 0.93, max pert 0.50, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scene with strong deceleration due to pedestrians.
    
    - Deceleration to maintain a safe distance from pedestrians.
    - Trajectory should show a speed drop of at least 2.9 m/s.
    - Perceptual mention of pedestrians.
    """
    comp = {
        "perceptual_pedestrian": 0.0,
        "decelerate_commitment": 0.0,
        "speed_drop_execution": 0.0,
    }
    
    # Perceptual mention of pedestrians
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        
        # Graded speed drop execution
        if speed_drop > 0:
            comp["speed_drop_execution"] = 0.5 * min(1.0, speed_drop / 5.8)
        
        # Deceleration commitment credit
        comp["decelerate_commitment"] = 0.4

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
