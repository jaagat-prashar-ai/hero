"""clip d1ea4db7-302e-41d1-bf94-04a800dd62e9 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring a rollout based on decisive events:
    1. Pedestrian crossing: Expect deceleration to yield, with a speed drop of at least 5.0 m/s.
    """
    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "decelerate_execution": 0.0
    }

    # Check for perceptual mention of pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Check for deceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration execution
        comp["decelerate_execution"] = 0.9 * min(1.0, speed_drop / 10.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
