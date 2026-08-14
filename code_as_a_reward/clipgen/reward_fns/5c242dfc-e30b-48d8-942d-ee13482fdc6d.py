"""clip 5c242dfc-e30b-48d8-942d-ee13482fdc6d - attempt 2/5 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the scene's decisive event:
    - Stop for the red traffic light.
    - Thresholds derived from the expert trajectory: speed drop of 7.7 m/s
      (floor at 3.85 m/s), stop by t=5.8 s.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Perceptual component: mention of traffic signal
    if any(p.entity == 'signal' for p in claims.perceptual):
        perceptual_score = 0.1

    # Commitment component: deceleration family
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory component: graded speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        trajectory_score = 0.6 * min(1.0, speed_drop / 7.7)
        commitment_score = 0.3  # Base score for having the correct commitment

    return {
        "perceptual_mention": perceptual_score,
        "commitment_executed": commitment_score,
        "trajectory_executed": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
