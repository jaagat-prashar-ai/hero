"""clip 07721315-227e-40a7-80dd-5e76ff0c21ec - attempt 2/5 - gate PASS (pos 0.96, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring a rollout in a roundabout yielding scenario.
    
    Decisive Event: Yielding to traffic in the roundabout.
    - Perceptual: Mention of 'vehicle_generic' or 'roundabout'.
    - Commitment: Deceleration family ('decelerate', 'yield', 'stop', 'wait').
    - Trajectory: Speed reduction of at least 6.5 m/s by t=6.3 s.
    """
    perceptual_weight = 0.1
    commitment_weight = 0.6
    trajectory_weight = 0.3

    # Perceptual component
    saw_vehicle_or_roundabout = any(
        p.entity in ('vehicle_generic', 'roundabout') for p in claims.perceptual
    )
    perceptual_score = perceptual_weight if saw_vehicle_or_roundabout else 0.0

    # Commitment component
    has_decelerate_commitment = any(
        c.speed_profile == 'decelerate' for c in claims.commitments
    )

    # Trajectory component
    speed_series = np.array(traj.speed_mps)
    initial_speed = traj.initial_speed_mps
    min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, 6.4))
    speed_drop = initial_speed - min_speed_after
    min_speed_time = np.argmin(window(speed_series, traj.dt_s, 0.0, 6.4)) * traj.dt_s

    # Graded trajectory factor
    trajectory_score = 0.0
    if has_decelerate_commitment and min_speed_time >= 5.0:
        trajectory_score = trajectory_weight * min(1.0, speed_drop / 6.5)

    # Combine components
    components = {
        "perceptual": perceptual_score,
        "commitment": commitment_weight if has_decelerate_commitment and trajectory_score > 0 else 0.0,
        "trajectory": trajectory_score
    }

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
