"""clip 79937d25-63d0-4e38-9848-c714dcab0328 - attempt 2/5 - gate PASS (pos 0.76, max pert 0.25, real rollout argmax 9)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Stop for pedestrian crossing: Expect a 'decelerate' commitment and a speed drop of at least 2.5 m/s.
    2. Rider on the left: Minimal impact, no specific maneuver required.
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_slowing": 0.0,
    }

    # Perceptual component for pedestrian
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.05  # Reduced weight

    # Commitment component for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["commitment_decelerate"] = 0.2

        # Trajectory component for slowing down, gated by commitment
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop > 2.5:  # Half of the GT speed drop of 5.0 m/s
            comp["trajectory_slowing"] = 0.55 * min(1.0, speed_drop / 5.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
