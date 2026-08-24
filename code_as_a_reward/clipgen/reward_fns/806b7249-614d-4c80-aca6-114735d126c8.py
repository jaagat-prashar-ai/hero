"""clip 806b7249-614d-4c80-aca6-114735d126c8 - attempt 2/5 - gate PASS (pos 0.75, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 806b7249-614d-4c80-aca6-114735d126c8:
    - Decelerate to maintain safe distance from pedestrians on the left.
    - Perceptual mentions of pedestrians and truck.
    - Trajectory thresholds: speed drop >= 0.6 m/s.
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "perceptual_truck": 0.0,
        "decelerate_for_pedestrians": 0.0,
    }

    # Perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.05

    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        comp["perceptual_truck"] = 0.05

    # Deceleration commitment and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 0.6:
            comp["decelerate_for_pedestrians"] = 0.7 * min(1.0, speed_drop / 1.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
