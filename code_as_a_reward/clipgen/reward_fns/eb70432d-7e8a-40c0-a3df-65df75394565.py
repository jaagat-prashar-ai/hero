"""clip eb70432d-7e8a-40c0-a3df-65df75394565 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.16, real rollout argmax 0)"""
def components(claims, traj):
    """Components for the scene with a pedestrian crossing and a stop sign.
    
    Decisive Events:
    1. Pedestrian Crossing: Expect deceleration in response to a pedestrian.
    2. Stop Sign at Intersection: Expect deceleration approaching the intersection.
    
    Trajectory thresholds:
    - Speed drop: at least 2.0 m/s (half of the expert's 4.5 m/s drop).
    - Timing: Deceleration should be evident throughout the window.
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "perceptual_intersection": 0.0,
        "decelerate_for_pedestrian": 0.0,
        "decelerate_for_intersection": 0.0,
    }

    # Check for perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    if any(p.entity == 'intersection' for p in claims.perceptual):
        comp["perceptual_intersection"] = 0.1

    # Check for commitment claims and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration
        deceleration_factor = 0.6 * min(1.0, speed_drop / 4.5)

        # Deceleration for pedestrian
        if any(p.entity == 'pedestrian' for p in claims.perceptual):
            comp["decelerate_for_pedestrian"] = deceleration_factor

        # Deceleration for intersection
        if any(p.entity == 'intersection' for p in claims.perceptual):
            comp["decelerate_for_intersection"] = deceleration_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
