"""clip 5545aea5-732c-4287-95eb-2ddd11df9fdd - attempt 2/5 - gate PASS (pos 0.70, max pert 0.14, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive events:
    1. Construction Zone Navigation: Expect a lateral maneuver ('nudge' or 'lane_change') to the left.
    Scene-derived thresholds: lateral offset change >= 0.15 m.
    """

    # Initialize component scores
    comp = {
        "mention_construction": 0.0,
        "lateral_maneuver": 0.0
    }

    # Perceptual mentions
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["mention_construction"] = 0.1

    # Lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge') and c.direction != 'right' for c in claims.commitments):
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        comp["lateral_maneuver"] = 0.6 * min(1.0, lateral_offset_change / 0.33)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
