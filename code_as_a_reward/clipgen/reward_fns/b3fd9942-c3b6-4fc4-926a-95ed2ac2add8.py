"""clip b3fd9942-c3b6-4fc4-926a-95ed2ac2add8 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene with decisive events: steering left to avoid construction zone.
    - Lateral maneuver: Expect a leftward heading change of at least 1 degree.
    - Perceptual mentions: construction zone, traffic barriers, or vehicles.
    """
    # Initialize component scores
    perceptual_mention = 0.0
    lateral_maneuver = 0.0

    # Perceptual mention component
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'vehicle_generic') for p in claims.perceptual):
        perceptual_mention = 0.05

    # Lateral maneuver component
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        lateral_maneuver = 0.65 * min(1.0, heading_change / 1.0)

    return {
        "perceptual_mention": perceptual_mention,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
