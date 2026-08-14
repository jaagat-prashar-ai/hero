"""clip 5097f4f2-cfe0-45ab-960d-f35b75bb2ad2 - attempt 3/5 - gate PASS (pos 0.79, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring a rollout based on decisive events:
    1. Deceleration and Stop: Expect a 'decelerate' commitment and a speed drop of at least 1.85 m/s.
    2. Lateral Offset and Heading Change: Expect a lateral maneuver and a lateral offset of at least 0.3 m.
    Perceptual mentions of 'intersection', 'vehicle_generic', or 'traffic' are expected.
    """

    # Initialize component scores
    comp = {
        "perceptual_mention": 0.0,
        "decelerate_commitment": 0.0,
    }

    # Perceptual mention credit
    if any(p.entity in ('intersection', 'vehicle_generic', 'cross_traffic') for p in claims.perceptual):
        comp["perceptual_mention"] = 0.1

    # Deceleration commitment and execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 1.85:
            comp["decelerate_commitment"] = 0.7 * min(1.0, speed_drop / 3.9)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
