"""clip 1af79a22-41e1-42e1-b80d-273a1f8ee8f7 - attempt 4/5 - gate PASS (pos 0.80, max pert 0.30, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Decisive Event 1: Pedestrian Crossing
      - Perceptual mention of pedestrian or crosswalk.
      - Commitment to decelerate (yield/stop/wait/decelerate).
      - Trajectory should show a speed drop of at least 2.2 m/s by the end of the window.
    """

    # Initialize component scores
    scores = {
        "perceptual_pedestrian": 0.0,
        "perceptual_crosswalk": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_decelerate": 0.0,
    }

    # Perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        scores["perceptual_pedestrian"] = 0.05

    if any(p.entity == 'crosswalk' for p in claims.perceptual):
        scores["perceptual_crosswalk"] = 0.05

    # Commitment claims
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        scores["commitment_decelerate"] = 0.2

        # Trajectory analysis conditioned on commitment
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        if speed_drop >= 2.2:
            scores["trajectory_decelerate"] = 0.5 * min(1.0, speed_drop / 4.4)

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
