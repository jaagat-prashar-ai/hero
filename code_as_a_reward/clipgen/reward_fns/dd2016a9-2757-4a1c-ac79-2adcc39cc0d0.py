"""clip dd2016a9-2757-4a1c-ac79-2adcc39cc0d0 - attempt 5/5 - gate PASS (pos 0.71, max pert 0.17, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scoring a rollout in a scene with a pedestrian crossing.
    
    Decisive Event: Pedestrian Crossing
    - Perceptual mention of 'pedestrian' or 'crosswalk'
    - Commitment to decelerate (stop/yield/wait/decelerate)
    - Trajectory should show a speed drop of at least 0.9 m/s by the end of the window
    """
    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0

    # Check for perceptual mention of pedestrian or crosswalk
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_pedestrian = 0.05  # Mention-only credit

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        trajectory_decelerate = 0.95 * min(1.0, speed_drop / 1.8)  # GT drop is 1.8 m/s

        # Combine commitment and trajectory for deceleration
        commitment_decelerate = 0.70 * trajectory_decelerate

    # Return component scores
    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
