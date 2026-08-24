"""clip 5c0b2864-ca4b-4cde-8dd8-e772381fe695 - attempt 4/5 - gate PASS (pos 0.74, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing.
    
    Decisive Events:
    1. Pedestrian Crossing: Requires deceleration to yield.
       - Perceptual entity: 'pedestrian'
       - Commitment family: 'decelerate'
       - Trajectory: Speed drop of at least 1.1 m/s by ~2.8s.
    """
    comp = {
        "saw_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0
    }

    # Perceptual claim for pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.05

    # Commitment claim for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory speed drop
        speed_series = np.array(traj.speed_mps)
        min_speed_idx = np.argmin(window(speed_series, traj.dt_s, 0.0, 6.4))
        min_speed_time = min_speed_idx * traj.dt_s
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if min_speed_time <= 2.8:
            comp["decelerate_for_pedestrian"] = 0.95 * min(1.0, speed_drop / 2.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
