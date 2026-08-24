"""clip 1946f1c3-8638-4648-8944-506e6bffc4df - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing.
    - Pedestrian crossing: Expect mention of 'pedestrian' and a 'decelerate' commitment.
    - Trajectory: Expect speed drop of at least 1.7 m/s, graded above this floor, occurring after t=3.7s.
    """
    components = {}

    # Perceptual mention of pedestrian
    components['mention_pedestrian'] = 0.05 if any(
        p.entity in ('pedestrian',) for p in claims.perceptual) else 0.0

    # Commitment to decelerate for pedestrian
    slowing_pedestrian = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
    components['decelerate_for_pedestrian'] = (
        0.65 * min(1.0, speed_drop / 3.4) if slowing_pedestrian and min_speed_time > 3.7 else 0.0
    )

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
