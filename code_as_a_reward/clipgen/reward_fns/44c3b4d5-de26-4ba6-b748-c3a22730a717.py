"""clip 44c3b4d5-de26-4ba6-b748-c3a22730a717 - attempt 1/5 - gate PASS (pos 0.86, max pert 0.36, real rollout argmax 6)"""
def components(claims, traj):
    """Components for scene 44c3b4d5-de26-4ba6-b748-c3a22730a717:
    - Steering left through construction zone: Expect mention of construction-related entities
      and a lateral maneuver commitment (lane_change/nudge/merge/turn/enter/exit) excluding 'right'.
      Trajectory should show a significant leftward heading change, graded with a floor at ~28.65 deg.
    - Speed maintenance: No specific commitment required, but trajectory should show a speed increase,
      graded with a floor at ~1.9 m/s.
    """
    comp = {}

    # Perceptual mention of construction-related entities
    construction_mention = any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers')
                               for p in claims.perceptual)
    comp['construction_mention'] = 0.1 if construction_mention else 0.0

    # Lateral maneuver commitment (leftward)
    lateral_commitment = any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and
                             c.direction != 'right' for c in claims.commitments)
    heading_change = traj.total_heading_change_deg
    comp['lateral_execution'] = 0.5 * min(1.0, heading_change / 57.3) if lateral_commitment else 0.0

    # Speed maintenance (increase)
    speed_increase = traj.final_speed_mps - traj.initial_speed_mps
    comp['speed_increase'] = 0.3 * min(1.0, speed_increase / 3.8)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
