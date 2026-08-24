"""clip 17df9e01-be54-46b2-b1f1-39954539dd42 - attempt 1/5 - gate PASS (pos 0.80, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with construction vehicle and workers on the left.
    Decisive events:
    1. Steer right to maintain a safe distance from construction vehicle and workers.
       - Perceptual mention: {'vehicle_generic', 'workers'}
       - Commitment: Lateral maneuver (nudge/lane_change) to the right
       - Trajectory: Rightward lateral offset increase of at least +0.7 m
    2. Presence of other vehicles does not require additional maneuvers.
    """
    comp = {}

    # Perceptual component for construction vehicle and workers
    comp['perceptual_construction'] = 0.1 * any(
        p.entity in {'vehicle_generic', 'workers'} for p in claims.perceptual
    )

    # Lateral maneuver component
    lateral_maneuver_claimed = any(
        c.maneuver in ('lane_change', 'nudge', 'merge') and c.direction != 'left'
        for c in claims.commitments
    )
    lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
    comp['lateral_maneuver'] = 0.6 * lateral_maneuver_claimed * min(1.0, max(0.0, lateral_offset_change / 1.45))

    # Speed maintenance component (no significant speed change expected)
    speed_change = traj.final_speed_mps - traj.initial_speed_mps
    comp['speed_maintenance'] = 0.2 * (abs(speed_change) < 2.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
