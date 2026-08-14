"""clip 84163157-18dd-42f6-9f49-814b218d46a8 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scoring a rollout based on yielding to pedestrians:
    - Perceptual mention of pedestrians.
    - Commitment to decelerate (speed_profile='decelerate') with matching trajectory execution.
    - Trajectory showing a speed drop of at least 2.1 m/s, graded.
    """
    perceptual_weight = 0.1
    commitment_weight = 0.6
    trajectory_weight = 0.3

    # Perceptual component: mention of pedestrians
    saw_pedestrian = any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual)
    perceptual_score = perceptual_weight if saw_pedestrian else 0.0

    # Commitment component: deceleration with trajectory execution
    committed_to_decelerate = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    initial_speed = traj.initial_speed_mps
    min_speed = traj.min_speed_mps
    speed_drop = initial_speed - min_speed

    # Graded trajectory factor, floor at half the GT's speed drop
    trajectory_score = 0.0
    if committed_to_decelerate and speed_drop >= 2.1:
        trajectory_score = trajectory_weight * min(1.0, speed_drop / 4.2)

    commitment_score = commitment_weight if committed_to_decelerate and speed_drop >= 2.1 else 0.0

    return {
        "saw_pedestrian": perceptual_score,
        "commitment_to_decelerate": commitment_score,
        "trajectory_deceleration": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
