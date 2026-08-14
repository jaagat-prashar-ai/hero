"""clip 174d3445-a3cc-4699-8392-5a0cc728f8ff - attempt 1/5 - gate PASS (pos 0.90, max pert 0.13, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene 174d3445-a3cc-4699-8392-5a0cc728f8ff:
    - Steering left through construction zone: expect a leftward lateral offset increase of at least +3.4 meters.
    - Proximity to nearby vehicles: expect a leftward lateral offset increase early in the window.
    - Perceptual mentions of construction zone or vehicles.
    """
    comp = {
        "perceptual_construction_zone": 0.0,
        "lateral_left_construction": 0.0,
        "perceptual_vehicle": 0.0,
        "lateral_left_vehicle": 0.0,
    }

    # Perceptual mention of construction zone
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction_zone"] = 0.1

    # Lateral maneuver through construction zone
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        if lateral_offset_change > 0:
            comp["lateral_left_construction"] = 0.5 * min(1.0, lateral_offset_change / 6.74)

    # Perceptual mention of nearby vehicles
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.1

    # Lateral maneuver in response to nearby vehicles
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        if lateral_offset_change > 0:
            comp["lateral_left_vehicle"] = 0.3 * min(1.0, lateral_offset_change / 3.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
