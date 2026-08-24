"""clip 692f98ea-9611-4717-9082-25a303af3b2e - attempt 3/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scene with two decisive events:
    1. Oncoming vehicle avoidance: steer right to maintain distance.
       - Perceptual: 'oncoming_traffic', 'vehicle_generic'
       - Commitment: lateral maneuver (nudge/lane_change/merge) excluding left
       - Trajectory: rightward lateral offset change >= 2.0 m
    """

    # Initialize component scores
    comp = {
        "perceptual_vehicle": 0.0,
        "lateral_maneuver": 0.0,
    }

    # Perceptual components
    if any(p.entity in ('oncoming_traffic', 'vehicle_generic') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.1

    # Lateral maneuver component
    if any(c.maneuver in ('lane_change', 'nudge', 'merge') and c.direction != 'left' for c in claims.commitments):
        lateral_offset_change = traj.final_lateral_offset_m - min(window(traj.lateral_offset_m, traj.dt_s, 0.0, 3.0))
        comp["lateral_maneuver"] = 0.9 * min(1.0, lateral_offset_change / 4.16)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
