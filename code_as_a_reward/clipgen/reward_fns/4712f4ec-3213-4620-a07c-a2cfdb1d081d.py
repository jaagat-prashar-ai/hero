"""clip 4712f4ec-3213-4620-a07c-a2cfdb1d081d - attempt 3/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 8)"""
def components(claims, traj):
    """
    Components for scene 4712f4ec-3213-4620-a07c-a2cfdb1d081d:
    - Decelerate behind lead vehicle at traffic light: speed drop >= 4.0 m/s
    - Perceptual mentions: 'lead_vehicle', 'vehicle_generic'
    """

    # Initialize component scores
    comp = {
        "decelerate_behind_vehicle": 0.0,
        "mention_lead_vehicle": 0.0,
        "mention_vehicle_generic": 0.0
    }

    # Perceptual mentions
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["mention_lead_vehicle"] = 0.05
    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        comp["mention_vehicle_generic"] = 0.05

    # Decelerate behind lead vehicle at traffic light
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 4.0:  # Adjusted threshold for real execution
            comp["decelerate_behind_vehicle"] = 0.7 * min(1.0, speed_drop / 8.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
