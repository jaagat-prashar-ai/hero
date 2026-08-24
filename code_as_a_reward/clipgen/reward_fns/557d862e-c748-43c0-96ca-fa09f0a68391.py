"""clip 557d862e-c748-43c0-96ca-fa09f0a68391 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with a pedestrian requiring a yield.
    Decisive event: yield to pedestrian (Track 19).
    Trajectory expectations: maintain near-zero speed throughout the window.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_yield = 0.0
    trajectory_yield = 0.0

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for commitment to yield (speed_profile='decelerate')
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Use stop event as trajectory condition for yielding
        if traj.stop_event:
            trajectory_yield = 0.5

        # Combine commitment and trajectory for yielding
        if trajectory_yield > 0:
            commitment_yield = 0.4

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_yield": commitment_yield,
        "trajectory_yield": trajectory_yield
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
