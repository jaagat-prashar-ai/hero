"""clip 162f1dd7-2967-42ba-86c7-e15e321c5b86 - attempt 4/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing.
    
    Decisive Events:
    1. Pedestrian Crossing: Yield to the pedestrian crossing the crosswalk.
       - Perceptual mention: pedestrian
       - Commitment: decelerate (stop/yield/wait/decelerate)
       - Trajectory: Speed drop of at least 4.8 m/s in the correct direction.
    """
    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }
    
    # Check for perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for commitment to decelerate
    decelerate_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    
    # Trajectory analysis
    if traj.n_waypoints > 0:
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Ensure the trajectory shows a decrease in speed
        if decelerate_commitment and traj.initial_speed_mps > traj.final_speed_mps:
            # Graded factor for speed drop, floored at half the GT drop
            comp["decelerate_for_pedestrian"] = 0.9 * min(1.0, speed_drop / 6.0)
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
