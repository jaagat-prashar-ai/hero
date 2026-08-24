"""clip b042984a-9915-4b06-93a1-fa710c45acb1 - attempt 2/5 - gate PASS (pos 0.72, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene b042984a-9915-4b06-93a1-fa710c45acb1:
    1. Deceleration to stop behind the lead vehicle.
       - Perceptual mention: lead_vehicle.
       - Commitment: speed_profile='decelerate'.
       - Trajectory: speed drop of at least 4.0 m/s, graded.
    2. Lane maintenance.
       - Perceptual mention: lane.
       - Trajectory: lateral offset within ±0.1 m.
    """
    comp = {
        "mention_lead_vehicle": 0.0,
        "mention_lane": 0.0,
        "decelerate_executed": 0.0,
    }

    # Perceptual mentions
    if any(p.entity == 'lead_vehicle' for p in claims.perceptual):
        comp["mention_lead_vehicle"] = 0.1

    if any(p.entity == 'lane' for p in claims.perceptual):
        comp["mention_lane"] = 0.1

    # Deceleration commitment and execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 4.0:
            comp["decelerate_executed"] = 0.7 * min(1.0, speed_drop / 8.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
