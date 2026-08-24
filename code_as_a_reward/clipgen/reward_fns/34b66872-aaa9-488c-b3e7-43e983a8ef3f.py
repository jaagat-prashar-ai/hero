"""clip 34b66872-aaa9-488c-b3e7-43e983a8ef3f - attempt 5/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene 34b66872-aaa9-488c-b3e7-43e983a8ef3f:
    - Steering right to follow barricades: expect a lateral maneuver commitment
      with a significant rightward heading change.
    """
    lateral_weight = 1.0

    # Lateral maneuver component: any lateral maneuver excluding left
    lateral_commitment = any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments)
    heading_change = traj.total_heading_change_deg
    lateral_factor = 0.7 * min(1.0, abs(heading_change) / 4.0) if lateral_commitment and heading_change < 0 else 0.0
    lateral_score = lateral_weight * lateral_factor

    return {
        "lateral_maneuver": lateral_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
