"""clip d886054b-532b-4097-aa6a-6080f4650d43 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.16, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness in the scene where
    the ego vehicle needs to slightly shift left due to temporary traffic
    delineators ahead. The trajectory should show a lateral offset change
    of at least +0.17 m (half of the expert's +0.34 m) to the left.
    """

    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Check for perceptual claims related to traffic delineators
    if any(p.entity in ('construction_cones', 'barricades', 'work_zone') for p in claims.perceptual):
        perceptual_score = 0.1  # Small weight for mention

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor for lateral offset change
        if lateral_offset_change < 0:  # Ensure the shift is to the left
            lateral_maneuver_score = 0.7 * min(1.0, abs(lateral_offset_change) / 8.55)

    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver_executed": lateral_maneuver_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
