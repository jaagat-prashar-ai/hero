"""clip d2f77ac9-de8b-45b8-a258-838744a0c6f9 - attempt 3/5 - gate PASS (pos 0.80, max pert 0.12, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene with deceleration for roundabout entry.
    
    Decisive events:
    1. Deceleration for roundabout entry: Expect speed_profile='decelerate' and mention of 'vehicle' or 'roundabout'.
       Trajectory should show a speed drop of at least 5.8 m/s, graded with the magnitude of the drop.
    """
    comp = {
        "mention_vehicle_or_roundabout": 0.1,
        "deceleration_executed": 0.0,
    }

    # Perceptual mention credit
    if any(p.entity in ('vehicle_generic', 'roundabout') for p in claims.perceptual):
        comp["mention_vehicle_or_roundabout"] = 0.1

    # Commitment and trajectory execution for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        comp["deceleration_executed"] = 0.7 * min(1.0, speed_drop / 11.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
