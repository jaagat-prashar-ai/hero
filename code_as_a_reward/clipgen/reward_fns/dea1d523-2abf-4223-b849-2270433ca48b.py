"""clip dea1d523-2abf-4223-b849-2270433ca48b - attempt 4/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    1. Steering Right for Traffic Delineators: Expect a 'nudge' right with a lateral offset of at least -2.44 m.
    Perceptual mentions are small additive credits for relevant entities.
    """

    # Initialize component scores
    comp = {
        "nudge_right": 0.0,
    }

    # Lateral maneuver: nudge right
    if any(c.maneuver in ('nudge', 'lane_change', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        final_offset = traj.final_lateral_offset_m
        if final_offset < -2.44:  # Expect a rightward nudge
            comp["nudge_right"] = 0.7 * min(1.0, abs(final_offset) / 4.88)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
