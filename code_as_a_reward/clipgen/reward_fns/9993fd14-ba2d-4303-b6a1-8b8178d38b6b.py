"""clip 9993fd14-ba2d-4303-b6a1-8b8178d38b6b - attempt 2/5 - gate PASS (pos 0.90, max pert 0.40, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring rollouts in a scene with a pedestrian crossing and nearby automobiles.
    
    Decisive Events:
    1. Yield to pedestrian crossing the road.
       - Perceptual: Mention of 'pedestrian'.
       - Commitment: 'decelerate' family (stop/yield/wait/decelerate).
       - Trajectory: Speed drop of at least 3.0 m/s, ideally around t=3.9s.
    
    2. Deceleration in response to nearby automobiles.
       - Perceptual: Mention of 'vehicle_generic'.
       - Commitment: 'decelerate' family (stop/yield/wait/decelerate).
       - Trajectory: Speed drop of at least 3.0 m/s, ideally starting early in the window.
    """
    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "commitment_slowing": 0.0,
        "trajectory_slowing": 0.0
    }

    # Perceptual components
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.05  # Reduced weight for mentioning pedestrian

    # Commitment component for slowing
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["commitment_slowing"] = 0.35  # Increased weight for commitment to slow

        # Trajectory component for slowing, gated by commitment
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop > 3.0:  # Floor at half the GT drop
            comp["trajectory_slowing"] = 0.5 * min(1.0, speed_drop / 6.0)  # Graded factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
