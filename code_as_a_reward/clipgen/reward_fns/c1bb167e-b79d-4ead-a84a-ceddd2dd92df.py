"""clip c1bb167e-b79d-4ead-a84a-ceddd2dd92df - attempt 2/5 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene c1bb167e-b79d-4ead-a84a-ceddd2dd92df:
    - Steering left through construction zone: expect a leftward heading change of at least -24.5 degrees.
    - Speed maintenance: expect no significant speed drop.
    - Perceptual mentions: construction-related entities.
    """
    comp = {
        "perceptual_construction": 0.0,
        "lateral_steering": 0.0,
        "speed_maintenance": 0.0,
    }

    # Perceptual mention of construction-related entities
    if any(p.entity in ('construction_cones', 'work_zone', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.1

    # Lateral steering left through construction zone
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change >= 24.5:  # Expecting a positive heading change for leftward steering
            comp["lateral_steering"] = 0.6 * min(1.0, heading_change / 49.0)

    # Speed maintenance: no significant drop expected
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    if speed_drop <= 0.3:  # Expecting minimal speed drop
        comp["speed_maintenance"] = 0.3 * (1.0 - min(1.0, speed_drop / 0.3))

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
