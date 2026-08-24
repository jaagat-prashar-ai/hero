"""clip 989065d1-6f92-402d-b4af-0f29ac859093 - attempt 2/5 - gate PASS (pos 0.75, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 989065d1-6f92-402d-b4af-0f29ac859093:
    - Decisive Event 1: Pedestrian darting out, requiring a leftward maneuver.
      - Commitment: Lateral maneuver (nudge/lane_change) to the left
      - Trajectory: Leftward lateral offset increase, heading change
    - Decisive Event 2: Automobile on the left constraining the maneuver.
      - No separate commitment required, but lateral offset constraint
    """
    comp = {}

    # Lateral maneuver commitment
    lateral_commitment = any(
        c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right'
        for c in claims.commitments
    )
    
    # Trajectory expectations for lateral maneuver
    lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
    heading_change = traj.total_heading_change_deg

    # Graded lateral factor
    lateral_factor = 0.6 * min(1.0, lateral_offset_change / 0.6) if lateral_commitment else 0.0
    heading_factor = 0.4 * min(1.0, heading_change / 3.7) if lateral_commitment else 0.0

    comp['lateral_maneuver'] = lateral_factor
    comp['heading_change'] = heading_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
