"""clip f0947d1e-482a-44c9-94f1-3bdff177dc8c - attempt 1/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of yielding to a pedestrian at a crosswalk.
    Scene-derived thresholds:
    - Perceptual mention of pedestrian or crosswalk.
    - Commitment to decelerate (speed_profile='decelerate').
    - Trajectory speed reduction of at least 1.4 m/s (half of GT's 2.8 m/s drop), graded.
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check for perceptual mentions of pedestrian or crosswalk
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_score = 0.1  # Small additive weight for mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop over the trajectory
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for speed reduction
        trajectory_score = 0.5 * min(1.0, speed_drop / 2.8)

        # Combine with commitment check
        commitment_score = 0.4 * (trajectory_score > 0.0)

    # Return the component contributions
    return {
        "perceptual_mention": perceptual_score,
        "commitment_execution": commitment_score,
        "trajectory_execution": trajectory_score
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
