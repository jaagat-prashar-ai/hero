"""clip c83d0fd0-f4a7-4cd3-a170-d1a7b1e8ff0e - attempt 3/5 - gate PASS (pos 1.00, max pert 0.15, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Lateral Adjustment: Expect a leftward nudge or lane change with a lateral offset of at least +4.75 m (half of the positive case's +9.51 m) when a lateral maneuver is committed.
    - Perceptual Mention: Small credit for mentioning entities related to temporary traffic control (e.g., 'construction_cones', 'barricades').
    """

    # Initialize component scores
    lateral_adjustment_score = 0.0
    perceptual_mention_score = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('construction_cones', 'barricades', 'work_zone') for p in claims.perceptual):
        perceptual_mention_score = 0.1

    # Check for lateral maneuver commitment and corresponding trajectory
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        lateral_offset = traj.final_lateral_offset_m
        lateral_adjustment_score = 0.9 * min(1.0, lateral_offset / 9.51)  # Graded factor, floor at 4.75 m

    return {
        "lateral_adjustment": lateral_adjustment_score,
        "perceptual_mention": perceptual_mention_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
