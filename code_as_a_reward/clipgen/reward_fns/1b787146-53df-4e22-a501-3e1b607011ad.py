"""clip 1b787146-53df-4e22-a501-3e1b607011ad - attempt 5/5 - gate PASS (pos 0.74, max pert 0.05, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scoring a rollout based on the decisive event of yielding to a pedestrian crossing.
    
    Decisive Event: Pedestrian Crossing
    - Perceptual mention of 'pedestrian' or 'crosswalk'.
    - Commitment to 'decelerate' (family includes stop, yield, wait, decelerate).
    - Trajectory should show a speed drop of at least 1.45 m/s by around t=5.3 seconds.
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0

    # Check for perceptual mention of pedestrian or crosswalk
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_score = 0.05  # Small additive weight for perceptual mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Check if the minimum speed occurs at the expected time
        min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints))
        if min_speed_time >= 5.0:  # Ensure the deceleration happens towards the end of the window
            # Graded trajectory factor for deceleration
            trajectory_score = 0.70 * min(1.0, speed_drop / 2.9)  # Floor at 1.45 m/s drop
            # Combine commitment and trajectory scores
            commitment_score = trajectory_score

    # Return component contributions
    return {
        "perceptual_mention": perceptual_score,
        "deceleration_commitment": commitment_score
    }

def reward(claims, traj):
    # Clamp the sum of components to [0, 1]
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
