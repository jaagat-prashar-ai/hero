"""clip a1b95bea-6a47-4ff8-9209-3a70686eb1a3 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.12, real rollout argmax 9)"""
def components(claims, traj):
    """Components for scene a1b95bea-6a47-4ff8-9209-3a70686eb1a3:
    - Decisive Event 1: Steer left to pass the construction zone delineated by traffic barriers.
      - Perceptual: {'work_zone', 'construction_cones', 'barricades'}
      - Commitment: Lateral maneuver in {'lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit'}, excluding 'right'
      - Trajectory: Leftward lateral offset, final at least -1.2 m
    """
    comp = {}

    # Perceptual components
    comp['saw_construction_zone'] = 0.1 * any(
        p.entity in {'work_zone', 'construction_cones', 'barricades'}
        for p in claims.perceptual
    )

    # Commitment and trajectory components
    lateral_maneuver_claimed = any(
        c.maneuver in {'lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit'} and c.direction != 'right'
        for c in claims.commitments
    )

    if lateral_maneuver_claimed:
        final_offset = traj.final_lateral_offset_m
        comp['lateral_maneuver_executed'] = 0.7 * min(1.0, abs(final_offset) / 2.37)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
