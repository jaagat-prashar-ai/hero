"""clip e7f944bc-6f1c-4ac9-b004-dd54687d271d - attempt 2/5 - gate PASS (pos 0.90, max pert 0.05, real rollout argmax 9)"""
def components(claims, traj):
    """Decisive event: stopping behind the lead vehicle.
    Trajectory threshold: speed drop >= 2.65 m/s.
    Commitment credit matched at the FAMILY level (decelerate for stopping).
    Perceptual credit for mentioning 'lead_vehicle', 'lane', or 'intersection'.
    """

    # Initialize component scores
    comp = {
        "perceptual_lead_vehicle": 0.0,
        "perceptual_lane": 0.0,
        "perceptual_intersection": 0.0,
        "commitment_stop": 0.0
    }

    # Perceptual claims
    if any(p.entity == 'lead_vehicle' for p in claims.perceptual):
        comp["perceptual_lead_vehicle"] = 0.05
    if any(p.entity == 'lane' for p in claims.perceptual):
        comp["perceptual_lane"] = 0.05
    if any(p.entity == 'intersection' for p in claims.perceptual):
        comp["perceptual_intersection"] = 0.05

    # Commitment and trajectory for stopping
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 2.65:
            comp["commitment_stop"] = 0.85 * min(1.0, speed_drop / 5.3)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
