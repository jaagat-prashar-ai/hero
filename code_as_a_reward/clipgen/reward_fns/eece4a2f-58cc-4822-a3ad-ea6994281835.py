"""clip eece4a2f-58cc-4822-a3ad-ea6994281835 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.05, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene eece4a2f-58cc-4822-a3ad-ea6994281835:
    - Preparation to stop for a stop sign beyond the window.
    - Perceptual mention of 'intersection'.
    - Commitment to 'decelerate' for the stop sign.
    - Trajectory should show a significant speed drop.
    """
    # Initialize component scores
    perceptual_mention = 0.0
    commitment_decelerate = 0.0
    trajectory_execution = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('intersection',) for p in claims.perceptual):
        perceptual_mention = 0.05  # Reduced weight for mention-only credit

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop within the window
        initial_speed = traj.initial_speed_mps
        min_speed = min(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed

        # Graded trajectory factor for deceleration
        trajectory_execution = 0.7 * min(1.0, speed_drop / 1.65)  # Allow up to 3.3 m/s drop

        # Combine commitment and trajectory
        commitment_decelerate = 0.25 if trajectory_execution > 0 else 0.0

    return {
        "perceptual_mention": perceptual_mention,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_execution": trajectory_execution
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
