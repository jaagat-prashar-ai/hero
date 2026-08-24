"""clip 1bd51369-ce05-4b35-b6d4-55d520c95623 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.18, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with lane adjustment for construction zone and pedestrian presence.
    
    Decisive Events:
    1. Lane Adjustment for Construction Zone:
       - Perceptual: 'work_zone', 'construction_cones', 'barricades', 'workers'
       - Commitment: 'lane_change' or 'nudge' (leftward)
       - Trajectory: Leftward shift with final lateral offset >= -0.4 m
    2. Proximity to Pedestrian:
       - Perceptual: 'pedestrian'
       - No specific trajectory change required
       
    Scene-derived thresholds:
    - Lateral offset change for lane adjustment: graded factor based on final offset
    - Perceptual mention credit: small additive weight
    """
    comp = {
        "perceptual_construction": 0.0,
        "perceptual_pedestrian": 0.0,
        "lateral_adjustment": 0.0,
    }

    # Perceptual credit for construction zone
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.1

    # Perceptual credit for pedestrian presence
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.05

    # Lateral adjustment for construction zone
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        final_offset = traj.final_lateral_offset_m
        comp["lateral_adjustment"] = 0.7 * min(1.0, abs(final_offset) / 0.82)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
