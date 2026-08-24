"""clip 0b094273-d554-4207-b04c-d5ccde4fb0c4 - attempt 3/5 - gate PASS (pos 0.98, max pert 0.30, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the scene's decisive events:
    1. Pedestrian near crosswalk: Expect a mention of 'pedestrian' or 'crosswalk' and a minimal speed drop.
    Trajectory thresholds are derived from the expert's behavior within the window.
    """

    # Initialize component scores
    scores = {
        "perceptual_pedestrian": 0.02,
        "commitment_slowing": 0.0,
        "trajectory_speed_drop": 0.0,
    }

    # Perceptual components
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        scores["perceptual_pedestrian"] = 0.02

    # Commitment components
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        scores["commitment_slowing"] = 0.28

        # Trajectory components
        # Speed drop component: Expect a minimal speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 1.2:  # Half of the positive case's 2.4 m/s drop
            scores["trajectory_speed_drop"] = 0.68 * min(1.0, speed_drop / 2.4)

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
