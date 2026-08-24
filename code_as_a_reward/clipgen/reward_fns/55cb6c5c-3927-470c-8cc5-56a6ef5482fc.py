"""clip 55cb6c5c-3927-470c-8cc5-56a6ef5482fc - attempt 3/5 - gate PASS (pos 0.98, max pert 0.30, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene with road curve and temporary traffic barrel guidance.
    
    Decisive events:
    - Navigating the road curve guided by a temporary traffic barrel.
    
    Scene-derived thresholds:
    - Heading change: at least +3 degrees (half of +6 degrees).
    - Speed maintenance: speed drop of at least 0.9 m/s (half of 1.8 m/s).
    """
    lateral_weight = 0.30
    speed_maintenance_weight = 0.70

    # Lateral maneuver check
    lateral_maneuver = any(
        c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit')
        for c in claims.commitments
    )

    # Speed profile check
    slowing_commitment = any(
        c.speed_profile == 'decelerate' for c in claims.commitments
    )

    # Trajectory checks
    heading_change = traj.total_heading_change_deg
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps

    # Lateral maneuver execution
    lateral_execution = 0.0
    if lateral_maneuver:
        lateral_execution = lateral_weight * min(1.0, heading_change / 6.0)

    # Speed maintenance execution
    speed_maintenance_execution = 0.0
    if slowing_commitment:
        speed_maintenance_execution = speed_maintenance_weight * min(1.0, speed_drop / 1.8)

    return {
        "lateral_execution": lateral_execution,
        "speed_maintenance_execution": speed_maintenance_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
