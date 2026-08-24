"""clip 573b3533-2fd3-4582-a05f-de456a0332fc - attempt 2/5 - gate PASS (pos 0.74, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with slowing down for a pedestrian:
    - Perceptual mention of a pedestrian.
    - Commitment to decelerate (speed_profile='decelerate').
    - Trajectory showing a speed drop of at least 1.5 m/s.
    """
    comp = {
        "perceptual_pedestrian": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_slowing": 0.0,
    }

    # Perceptual mention of a pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory showing a speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 1.5:
            comp["commitment_decelerate"] = 0.2
            comp["trajectory_slowing"] = 0.5 * min(1.0, speed_drop / 3.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
