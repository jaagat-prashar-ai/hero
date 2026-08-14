"""clip e0f45fdc-1612-464f-8174-1212425ad8df - attempt 1/5 - gate PASS (pos 0.85, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with deceleration for traffic light and yielding to emergency vehicle and pedestrian.
    
    - Deceleration for traffic light: Expect perceptual mention of 'signal' or 'intersection' and a 'decelerate' commitment.
      Trajectory should show a speed drop of at least 3.5 m/s, graded.
    - Yielding to emergency vehicle and pedestrian: Expect perceptual mention of 'emergency_vehicle' or 'pedestrian' and a 'decelerate' commitment.
      Trajectory should show a speed drop of at least 2.0 m/s, graded.
    """
    comp = {}

    # Perceptual mentions
    comp['mention_signal_intersection'] = 0.05 * any(
        p.entity in ('signal', 'intersection') for p in claims.perceptual
    )
    comp['mention_emergency_pedestrian'] = 0.05 * any(
        p.entity in ('emergency_vehicle', 'pedestrian') for p in claims.perceptual
    )

    # Deceleration for traffic light
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    decelerate_claim = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    comp['decelerate_for_light'] = (
        0.5 * min(1.0, speed_drop / 7.0) if decelerate_claim and speed_drop >= 3.5 else 0.0
    )

    # Yielding to emergency vehicle and pedestrian
    comp['yield_to_emergency_pedestrian'] = (
        0.3 * min(1.0, speed_drop / 4.0) if decelerate_claim and speed_drop >= 2.0 else 0.0
    )

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
