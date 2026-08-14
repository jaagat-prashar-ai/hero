"""clip 8843953f-cec8-43fe-9eba-4fa20afe5999 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.25, real rollout argmax 6)"""
def components(claims, traj):
    """Components for scoring a rollout based on decisive events:
    1. Yielding to a pedestrian crossing the road.
       - Perceptual mention: 'pedestrian'
       - Commitment: 'decelerate' family
       - Trajectory: speed drop of at least 0.65 m/s by t=4.0 s
    """

    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "yield_to_pedestrian": 0.0
    }

    # Check for perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Check for commitment to yield (decelerate family)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop
        comp["yield_to_pedestrian"] = 0.65 * min(1.0, speed_drop / 1.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
