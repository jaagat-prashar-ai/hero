"""clip 645cc0d3-a688-4bdf-bc7b-e538f90c28bb - attempt 5/5 - gate PASS (pos 1.00, max pert 0.00, real rollout argmax 10)"""
def components(claims, traj):
    """Components for reward calculation based on decisive events:
    - Curved Road Navigation: Expect a rightward steering maneuver.
      - Lateral maneuver commitment (turn/nudge) excluding 'left'.
      - Trajectory should show a rightward heading change of at least -65 degrees.
    """
    lateral_weight = 1.0  # Full weight allocated to lateral execution

    # Lateral maneuver commitment check
    lateral_commitment = any(c.maneuver in ('turn', 'nudge', 'lane_change', 'merge', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments)

    # Trajectory heading change check
    heading_change = traj.total_heading_change_deg
    heading_change_score = 0.0
    if lateral_commitment:
        heading_change_floor = -65.0
        heading_change_score = lateral_weight * min(1.0, abs(heading_change) / 77.0) if heading_change <= heading_change_floor else 0.0

    return {
        "lateral_execution": heading_change_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
