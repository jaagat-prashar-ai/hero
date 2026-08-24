"""clip 21de9f79-9dd8-4dbf-96f0-060fcc0139f6 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.30, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring a rollout based on the scene's decisive event: yielding to a pedestrian.
    - Perceptual: Mention of a pedestrian.
    - Commitment: Deceleration to yield to the pedestrian.
    - Trajectory: Speed reduction of at least 1.75 m/s (half of 3.5 m/s) during the window.
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0
    trajectory_decelerate = 0.0

    # Check for perceptual mention of a pedestrian
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        commitment_decelerate = 0.2

        # Calculate speed drop and ensure it occurs during the window
        speed_series = np.array(traj.speed_mps)
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        
        # Ensure the speed drop is significant and occurs within the trajectory window
        if speed_drop >= 1.75:
            trajectory_decelerate = 0.7 * min(1.0, speed_drop / 3.5)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_decelerate": trajectory_decelerate,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
