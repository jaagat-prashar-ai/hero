"""clip 6b3e802e-ea88-48c8-aa13-3e072c8c1409 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring a rollout based on the scene's decisive events:
    - Strong deceleration to yield to a pedestrian crossing the road.
    - Trajectory should show a speed drop of at least 3.35 m/s (half of GT's 6.7 m/s).
    - Perceptual mention of a pedestrian entity.
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "decelerate_commitment": 0.0,
        "trajectory_deceleration": 0.0
    }

    # Perceptual component: mention of pedestrian
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Commitment component: deceleration family
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory component: graded deceleration
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        comp["trajectory_deceleration"] = 0.5 * min(1.0, speed_drop / 6.7)
        comp["decelerate_commitment"] = 0.4

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
