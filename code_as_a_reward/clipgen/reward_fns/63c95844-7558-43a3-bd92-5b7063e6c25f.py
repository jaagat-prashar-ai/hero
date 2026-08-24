"""clip 63c95844-7558-43a3-bd92-5b7063e6c25f - attempt 1/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 4)"""
def components(claims, traj):
    """
    Components for scene 63c95844-7558-43a3-bd92-5b7063e6c25f:
    - Deceleration for pedestrians: perceptual mention of 'pedestrian',
      commitment to 'decelerate', and a graded speed drop.
    - Trajectory expectations: speed drop of at least 0.7 m/s, graded
      with higher scores for greater deceleration.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0
    speed_drop_score = 0.0

    # Check for perceptual mention of pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded speed drop score
        speed_drop_score = 0.5 * min(1.0, speed_drop / 1.4)

        # Combine with commitment
        commitment_decelerate = 0.4 * (0.5 * min(1.0, speed_drop / 1.4))

    # Return component scores
    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
        "speed_drop_score": speed_drop_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
