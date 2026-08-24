"""clip 150fa0f1-d414-4726-91f8-9de29bbfa63a - attempt 1/5 - gate PASS (pos 0.70, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing and nearby automobiles.
    
    Decisive Events:
    1. Pedestrian Crossing: Expect mention of 'pedestrian' and a deceleration
       commitment. Trajectory should show a speed drop of at least 0.5 m/s.
    2. Automobiles on the Right: Mention of 'vehicle_generic' is expected, but
       no specific commitment is required. Minimal lateral offset change.
       
    Trajectory thresholds are based on approximately half the ground truth
    magnitudes, with graded factors for speed drop.
    """
    # Initialize component scores
    scores = {
        "saw_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
        "saw_vehicles": 0.0,
        "maintained_lane": 0.0
    }
    
    # Check for pedestrian perception
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        scores["saw_pedestrian"] = 0.1  # Small weight for mention

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = traj.min_speed_mps
        speed_drop = initial_speed - min_speed_after
        
        # Graded factor for speed drop
        scores["decelerate_for_pedestrian"] = 0.5 * min(1.0, speed_drop / 1.0)

    # Check for vehicle perception
    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        scores["saw_vehicles"] = 0.1  # Small weight for mention

    # Check for lane maintenance (minimal lateral offset change)
    lateral_offset_change = abs(traj.final_lateral_offset_m)
    if lateral_offset_change <= 0.5:
        scores["maintained_lane"] = 0.1  # Minimal weight, as no specific maneuver required

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
