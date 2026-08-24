"""clip 4d1706d3-fc3f-4275-985c-1987a390015f - attempt 3/5 - gate PASS (pos 1.00, max pert 0.13, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scene with pedestrians and curving road.
    
    Decisive Events:
    1. Pedestrians crossing the crosswalk: Expect mention of pedestrians or crosswalk, commitment to decelerate, and a speed drop.
    
    Trajectory thresholds:
    - Speed drop: at least 3.8 m/s for deceleration.
    """
    # Initialize component scores
    components = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrians": 0.0,
    }

    # Perceptual mentions
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        components["mention_pedestrian"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after
        components["decelerate_for_pedestrians"] = 0.9 * min(1.0, speed_drop / 7.6)

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
