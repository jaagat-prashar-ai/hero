"""clip b7f37a71-fa59-4c38-850f-d09984d15b65 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    1. Perceptual mention of construction cones.
    2. Lateral maneuver to the left to avoid the construction cone.
    Scene-derived thresholds: lateral offset increase to at least +0.82 m.
    """

    # Initialize component scores
    perceptual_cones = 0.0
    lateral_maneuver_left = 0.0

    # Check for perceptual mention of construction cones
    if any(p.entity in ('construction_cones', 'work_zone', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_cones = 0.05  # Small mention-only credit

    # Check for lateral maneuver to the left
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the lateral offset increase
        final_lateral_offset = traj.final_lateral_offset_m
        if final_lateral_offset <= -0.82:  # Half of the measured -1.64 m
            lateral_maneuver_left = 0.65 * min(1.0, abs(final_lateral_offset) / 1.64)

    return {
        "perceptual_cones": perceptual_cones,
        "lateral_maneuver_left": lateral_maneuver_left,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
