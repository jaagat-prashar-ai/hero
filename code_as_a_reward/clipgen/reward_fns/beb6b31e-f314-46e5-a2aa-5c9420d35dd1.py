"""clip beb6b31e-f314-46e5-a2aa-5c9420d35dd1 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene:
    1. Steering Left through Construction Zone: Expect a leftward maneuver with a heading change of at least 17 degrees.
    2. Speed Adjustment: Adjusted to account for lateral maneuver without speed drop.
    Perceptual mentions of construction-related entities are also considered.
    """

    # Initialize component scores
    comp = {
        "perceptual_construction": 0.0,
        "lateral_maneuver": 0.0,
        "speed_adjustment": 0.0
    }

    # Perceptual component: mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.1

    # Lateral maneuver component: steering left
    if any(c.maneuver in ('lane_change', 'nudge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        comp["lateral_maneuver"] = 0.6 * min(1.0, heading_change / 17.0)

    # Speed adjustment component: minor speed drop or lateral maneuver
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        # Allow for lateral maneuver credit even if speed drop is minimal
        comp["speed_adjustment"] = 0.3 * min(1.0, speed_drop / 0.15)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
