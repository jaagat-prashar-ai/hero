"""clip 7d874fe5-b316-4f2d-926c-3cb034ddf5d4 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene 7d874fe5-b316-4f2d-926c-3cb034ddf5d4:
    - Decelerate for oncoming vehicle: speed drop >= 2.7 m/s, commitment 'decelerate'
    - Maintain safe distance from worker: lateral offset >= 0.135 m, commitment 'nudge'
    """
    # Initialize component scores
    decelerate_component = 0.0
    nudge_component = 0.0
    perceptual_oncoming_vehicle = 0.0
    perceptual_worker = 0.0

    # Check for perceptual claims
    if any(p.entity in ('oncoming_traffic', 'vehicle_generic') for p in claims.perceptual):
        perceptual_oncoming_vehicle = 0.05

    if any(p.entity == 'workers' for p in claims.perceptual):
        perceptual_worker = 0.05

    # Check for deceleration commitment and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 2.7:
            decelerate_component = 0.6 * min(1.0, speed_drop / 5.4)

    # Check for nudge commitment and trajectory execution
    if any(c.maneuver in ('nudge', 'lane_change', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        lateral_offset = abs(traj.final_lateral_offset_m)
        if lateral_offset >= 0.135:
            nudge_component = 0.3 * min(1.0, lateral_offset / 0.27)

    return {
        "perceptual_oncoming_vehicle": perceptual_oncoming_vehicle,
        "perceptual_worker": perceptual_worker,
        "decelerate_component": decelerate_component,
        "nudge_component": nudge_component
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
