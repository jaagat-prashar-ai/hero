"""clip 59d21041-f675-4ad8-af57-ab1d76abe38b - attempt 5/5 - gate PASS (pos 1.00, max pert 0.30, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with decisive event: stopping behind the lead vehicle.
    - Perceptual: mention of a vehicle entity.
    - Commitment: deceleration family (stop/yield/wait/decelerate).
    - Trajectory: speed drop of at least 0.1 m/s by t=0.7 s.
    """
    perceptual_vehicle = any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual)
    commitment_decelerate = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Calculate speed drop
    initial_speed = traj.initial_speed_mps
    min_speed = traj.min_speed_mps
    speed_drop = initial_speed - min_speed

    # Determine if the speed drop is significant and occurs early
    speed_drop_factor = 0.7 * min(1.0, speed_drop / 0.1) if speed_drop >= 0.1 and np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s <= 0.7 else 0.0

    # Components
    components = {
        "perceptual_vehicle": 0.05 if perceptual_vehicle else 0.0,
        "commitment_decelerate": 0.25 if commitment_decelerate else 0.0,
        "trajectory_speed_drop": speed_drop_factor if commitment_decelerate else 0.0
    }

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
