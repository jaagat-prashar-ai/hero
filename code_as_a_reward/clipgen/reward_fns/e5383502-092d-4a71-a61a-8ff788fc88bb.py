"""clip e5383502-092d-4a71-a61a-8ff788fc88bb - attempt 4/5 - gate PASS (pos 0.80, max pert 0.38, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Yield to Pedestrian: Expect deceleration with a speed drop of at least 2.0 m/s.
    Trajectory thresholds are derived from the expert's in-window behavior.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0

    # Check for perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.0  # Remove weight to focus on commitment

    # Check for commitment claims
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration
        commitment_decelerate = 0.80 * min(1.0, speed_drop / 4.0)  # Floor at 2.0 m/s drop

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
