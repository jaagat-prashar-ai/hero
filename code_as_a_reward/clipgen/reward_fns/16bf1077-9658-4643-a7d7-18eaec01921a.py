"""clip 16bf1077-9658-4643-a7d7-18eaec01921a - attempt 2/5 - gate PASS (pos 0.95, max pert 0.05, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene with construction vehicle and pedestrian:
    - Steer left to avoid construction vehicle (lateral offset change >= 1.125 m)
    - Perceptual mentions for construction vehicle and pedestrian
    """
    comp = {
        "steer_left": 0.0,
        "mention_construction_vehicle": 0.0,
        "mention_pedestrian": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity in ('vehicle_generic', 'work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["mention_construction_vehicle"] = 0.05

    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Check for lateral maneuver to the left
    if any(c.maneuver in ('lane_change', 'nudge', 'merge') and c.direction != 'right' for c in claims.commitments):
        lateral_offset_change = abs(traj.final_lateral_offset_m)  # Assuming final offset is indicative of maneuver
        if lateral_offset_change >= 1.125:  # Half of the GT's max offset
            comp["steer_left"] = 0.9 * min(1.0, lateral_offset_change / 2.25)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
