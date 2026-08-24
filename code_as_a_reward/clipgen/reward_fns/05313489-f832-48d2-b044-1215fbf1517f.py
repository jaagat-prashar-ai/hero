"""clip 05313489-f832-48d2-b044-1215fbf1517f - attempt 2/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene 05313489-f832-48d2-b044-1215fbf1517f:
    - Steering right through a construction zone: expect a rightward heading change
      and lateral offset, with perceptual mentions of construction-related entities.
    """
    # Initialize component scores
    comp = {
        "perceptual_construction": 0.0,
        "lateral_maneuver": 0.0,
    }

    # Perceptual mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.1

    # Lateral maneuver: rightward steering
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change < 0:  # Expect rightward change
            comp["lateral_maneuver"] = 0.9 * min(1.0, abs(heading_change) / 18.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
