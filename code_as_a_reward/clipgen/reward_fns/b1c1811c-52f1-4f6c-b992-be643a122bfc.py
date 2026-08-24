"""clip b1c1811c-52f1-4f6c-b992-be643a122bfc - attempt 2/5 - gate PASS (pos 0.80, max pert 0.23, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene b1c1811c-52f1-4f6c-b992-be643a122bfc:
    - Lane change to the left while maintaining a safe distance from the yellow construction vehicle on the right.
    - Lateral offset change of at least +1.25 m to the left.
    - Perceptual mention of 'vehicle_generic' or 'construction' entities.
    """

    # Initialize component scores
    comp = {
        "perceptual_mention": 0.0,
        "lane_change_execution": 0.0,
    }

    # Perceptual mention component
    if any(p.entity in ('vehicle_generic', 'construction_cones', 'barricades', 'work_zone') for p in claims.perceptual):
        comp["perceptual_mention"] = 0.1

    # Lane change execution component
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor for lateral offset change
        comp["lane_change_execution"] = 0.7 * min(1.0, lateral_offset_change / 1.25)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
