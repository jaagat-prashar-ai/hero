"""clip 498037d9-bc6b-4f9d-ac01-6538083c6878 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with deceleration to maintain safe distance and curve navigation.
    
    Decisive Events:
    1. Deceleration to maintain a safe distance from the barrier.
       - Perceptual entity: {'vehicle_generic', 'barricades'}
       - Commitment family: speed_profile='decelerate'
       - Trajectory: Speed drop of at least 6.0 m/s, graded factor.
    """
    perceptual_weight = 0.1
    deceleration_weight = 0.8

    # Perceptual components
    saw_barrier = perceptual_weight * any(
        p.entity in {'vehicle_generic', 'barricades'} for p in claims.perceptual
    )
    saw_curve = perceptual_weight * any(
        p.entity in {'curve', 'lane'} for p in claims.perceptual
    )

    # Commitment components
    decelerate_commitment = any(
        c.speed_profile == 'decelerate' for c in claims.commitments
    )

    # Trajectory components
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
    deceleration_executed = (
        deceleration_weight * min(1.0, speed_drop / 12.0)
        if decelerate_commitment and min_speed_time > 3.0
        else 0.0
    )

    return {
        "saw_barrier": saw_barrier,
        "saw_curve": saw_curve,
        "deceleration_executed": deceleration_executed,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
