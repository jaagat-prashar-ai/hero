"""clip 1de484ec-e29f-41ca-b4e8-4c91e0a1d66c - attempt 1/5 - gate PASS (pos 0.94, max pert 0.14, real rollout argmax 10)"""
def components(claims, traj):
    """Components for scene with stopping and yielding events:
    - Stop for lead vehicle: speed drop >= 1.2 m/s, ideally 2.4 m/s, with 'decelerate' commitment.
    - Yield to pedestrian: maintain low speed, ideally 0.2 m/s, with 'decelerate' commitment.
    - Perceptual mentions: 'vehicle_generic' and 'pedestrian' for small additive credit.
    """
    # Initialize component scores
    comp = {
        "stop_for_vehicle": 0.0,
        "yield_to_pedestrian": 0.0,
        "mention_vehicle": 0.0,
        "mention_pedestrian": 0.0
    }

    # Check for perceptual mentions
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["mention_vehicle"] = 0.05

    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Check for commitment to decelerate
    decelerate_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Trajectory analysis for stopping
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    if decelerate_commitment:
        comp["stop_for_vehicle"] = 0.5 * min(1.0, speed_drop / 2.4)

    # Trajectory analysis for yielding
    if decelerate_commitment:
        low_speed_maintenance = max(0.0, 1.0 - traj.final_speed_mps / 2.4)
        comp["yield_to_pedestrian"] = 0.4 * low_speed_maintenance

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
