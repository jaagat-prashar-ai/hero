"""clip 9c33903c-1242-42a0-b9dd-a93fffcac0c8 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrians crossing.
    
    Decisive events:
    - Pedestrian crossing: Expect deceleration or speed maintenance.
    
    Trajectory thresholds:
    - Speed maintenance: Floor at 0.5 m/s drop.
    """
    comp = {
        "perceptual_pedestrian": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_speed_maintenance": 0.0,
    }
    
    # Perceptual component for pedestrian mention
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Commitment component for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory component for speed maintenance
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        comp["commitment_decelerate"] = 0.3
        comp["trajectory_speed_maintenance"] = 0.6 * min(1.0, speed_drop / 1.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
