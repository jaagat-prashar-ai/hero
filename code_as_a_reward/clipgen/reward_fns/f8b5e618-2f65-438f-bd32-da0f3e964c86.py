"""clip f8b5e618-2f65-438f-bd32-da0f3e964c86 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.11, real rollout argmax 0)"""
def components(claims, traj):
    """Components for navigating a construction zone with lane closures and construction equipment on the right.
    
    Decisive events:
    - Maintain speed while navigating through a construction zone.
    - Lateral maneuver to the left to avoid construction equipment on the right.
    
    Scene-derived thresholds:
    - Lateral offset increase of at least +6.4 m to the left.
    - Perceptual mention of construction-related entities.
    """
    comp = {
        "perceptual_mention": 0.0,
        "lateral_maneuver": 0.0,
    }

    # Perceptual mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_mention"] = 0.1

    # Lateral maneuver to the left
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        comp["lateral_maneuver"] = 0.6 * min(1.0, max(0.0, lateral_offset_change / 12.81))

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
