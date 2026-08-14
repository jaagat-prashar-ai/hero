"""clip c9026dc1-bcc0-4d68-90a4-80d834eb6ef1 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.13, real rollout argmax 10)"""
def components(claims, traj):
    """Components for reward function based on decisive events:
    1. Pedestrian crossing: Expect deceleration to maintain safe distance.
       - Perceptual: Mention of 'pedestrian'.
       - Commitment: Deceleration (speed_profile='decelerate').
       - Trajectory: Speed drop of at least 0.4 m/s, graded.
    """
    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_executed": 0.0,
    }

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded factor for deceleration execution
        comp["decelerate_executed"] = 0.6 * min(1.0, speed_drop / 0.8)

    return comp

def reward(claims, traj):
    # Sum components and clamp to [0, 1]
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
