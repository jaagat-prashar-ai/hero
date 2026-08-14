"""clip 8d575002-79a7-4ee7-8a98-bfb8fdb98abf - attempt 2/5 - gate PASS (pos 0.70, max pert 0.23, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing.
    
    Decisive Event:
    1. Pedestrian Crossing: Requires deceleration commitment and speed drop.
    
    Scene-derived thresholds:
    - Speed drop for pedestrian: at least 3.5 m/s.
    """
    # Initialize component scores
    comp = {
        "saw_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for perceptual claims
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.1

    # Check for commitment claims
    decelerate_claim = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Calculate trajectory-based factors
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    if decelerate_claim:
        comp["decelerate_for_pedestrian"] = 0.6 * min(1.0, speed_drop / 7.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
