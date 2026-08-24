"""clip 17f79293-f912-4a3a-a64c-4505a904645f - attempt 2/5 - gate PASS (pos 0.95, max pert 0.05, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene 17f79293-f912-4a3a-a64c-4505a904645f:
    1. Maintain speed and safe distance from cut-in cyclist.
       - Perceptual mention: cyclist
       - Commitment: decelerate (stop/yield/wait/decelerate)
       - Trajectory: speed drop >= 0.65 m/s, min speed around t=6.1s
    2. Maintain position relative to nearby vehicles.
       - Perceptual mention: vehicle_generic
       - Commitment: decelerate (stop/yield/wait/decelerate)
       - Trajectory: speed drop >= 0.65 m/s
    """

    # Initialize component scores
    comp = {
        "mention_cyclist": 0.0,
        "decelerate_for_cyclist": 0.0,
        "decelerate_for_vehicle": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity == 'cyclist' for p in claims.perceptual):
        comp["mention_cyclist"] = 0.05

    # Check for deceleration commitment
    has_decelerate_commitment = any(
        c.speed_profile == 'decelerate' for c in claims.commitments
    )

    # Trajectory analysis
    speed_series = np.array(traj.speed_mps)
    min_speed_idx = np.argmin(window(speed_series, traj.dt_s, 0, 6.4))
    min_speed_time = min_speed_idx * traj.dt_s
    initial_speed = traj.initial_speed_mps
    min_speed = traj.min_speed_mps
    speed_drop = initial_speed - min_speed

    # Deceleration for cyclist
    if has_decelerate_commitment and min_speed_time >= 5.5 and min_speed_time <= 6.5:
        comp["decelerate_for_cyclist"] = 0.5 * min(1.0, speed_drop / 1.3)

    # Deceleration for vehicle
    if has_decelerate_commitment:
        comp["decelerate_for_vehicle"] = 0.4 * min(1.0, speed_drop / 1.3)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
