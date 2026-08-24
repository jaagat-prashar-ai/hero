"""clip 08dfff23-109f-4df5-967f-61ee37f481d9 - attempt 4/5 - gate PASS (pos 0.90, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene with stopping/creeping behavior and steering left through a construction zone.
    
    Decisive Events:
    1. Stopping/Creeping Behavior: Expect mention of 'vehicle_generic' and commitment to 'decelerate'.
       Trajectory should show a speed drop of at least 0.15 m/s.
    2. Steering Left: Expect mention of 'construction_cones' or 'work_zone' and commitment to 'nudge'.
       Trajectory should show a lateral offset change of at least 0.015 m.
    """
    comp = {}

    # Perceptual mention of vehicles
    comp['mention_vehicle'] = 0.05 * any(
        p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle')
        for p in claims.perceptual
    )

    # Perceptual mention of construction zone
    comp['mention_construction'] = 0.05 * any(
        p.entity in ('construction_cones', 'work_zone', 'barricades', 'workers')
        for p in claims.perceptual
    )

    # Commitment to nudge left
    nudge_claim = any(
        c.maneuver in ('nudge', 'lane_change', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right'
        for c in claims.commitments
    )
    if nudge_claim:
        lateral_change = abs(traj.final_lateral_offset_m - traj.lateral_offset_m[0])
        total_turn = traj.total_heading_change_deg
        comp['nudge_left'] = 0.8 * min(1.0, lateral_change / 44.79) if total_turn > 0 else 0.0
    else:
        comp['nudge_left'] = 0.0

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
