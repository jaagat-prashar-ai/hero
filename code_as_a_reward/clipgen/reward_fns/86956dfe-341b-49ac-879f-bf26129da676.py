"""clip 86956dfe-341b-49ac-879f-bf26129da676 - attempt 2/5 - gate PASS (pos 0.76, max pert 0.10, real rollout argmax 7)"""
def components(claims, traj):
    """
    Components for scene with two decisive events:
    1. Maintaining distance from the lead vehicle.
    2. Navigating through the construction zone.
    
    Thresholds derived from expert trajectory:
    - Speed increase: GT 4.0 m/s, floor 2.0 m/s.
    """

    # Initialize component scores
    comp = {
        "perceptual_lead_vehicle": 0.0,
        "perceptual_construction_zone": 0.0,
        "maintain_distance_execution": 0.0
    }

    # Perceptual claims
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["perceptual_lead_vehicle"] = 0.1

    if any(p.entity in ('work_zone', 'construction_cones', 'barricades') for p in claims.perceptual):
        comp["perceptual_construction_zone"] = 0.1

    # Commitment and trajectory checks
    # Event 1: Maintaining distance from the lead vehicle
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        comp["maintain_distance_execution"] = 0.7 * min(1.0, speed_increase / 4.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
