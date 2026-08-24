"""clip 7becc4fe-82cf-4910-af9a-e04eab61a4e5 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.26, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 7becc4fe-82cf-4910-af9a-e04eab61a4e5:
    - Steering right to follow traffic barrels: lateral offset change
      rightward, perceptual mention of traffic barrels or similar.
    - Speed adjustment: speed drop, perceptual mention of vehicles or
      traffic conditions.
    Thresholds derived from GT: lateral offset change >= 2.5 m, speed drop >= 1.1 m/s.
    """

    # Initialize component scores
    comp = {
        "mention_traffic_barrels": 0.0,
        "lateral_maneuver": 0.0,
        "mention_vehicles": 0.0,
        "speed_adjustment": 0.0,
    }

    # Perceptual mention of traffic barrels or similar
    if any(p.entity in ('barricades', 'construction_cones', 'work_zone') for p in claims.perceptual):
        comp["mention_traffic_barrels"] = 0.1

    # Lateral maneuver to the right
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        lateral_offset_change = abs(traj.final_lateral_offset_m) - abs(traj.lateral_offset_m[0])
        comp["lateral_maneuver"] = 0.5 * min(1.0, lateral_offset_change / 5.0)

    # Perceptual mention of vehicles or traffic conditions
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        comp["mention_vehicles"] = 0.1

    # Speed adjustment (deceleration)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        comp["speed_adjustment"] = 0.3 * min(1.0, speed_drop / 2.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
