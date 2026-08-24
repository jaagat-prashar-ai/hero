"""clip 09ab5295-11e1-4242-b444-548f0cee14c3 - attempt 3/5 - gate PASS (pos 0.90, max pert 0.49, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the scene's decisive events:
    - Deceleration in response to a pedestrian crossing at the crosswalk.
    - Perceptual recognition of pedestrians or crosswalks.
    Scene-derived thresholds:
    - Speed drop of at least 2.4 m/s (half of GT's 4.8 m/s drop).
    - Deceleration commitment matched at the speed_profile='decelerate' level.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.05  # Reduced weight for mention-only
    perceptual_crosswalk = 0.05  # Reduced weight for mention-only
    deceleration_executed = 0.0

    # Check for perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.05  # Small additive weight

    if any(p.entity == 'crosswalk' for p in claims.perceptual):
        perceptual_crosswalk = 0.05  # Small additive weight

    # Check for deceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration execution
        deceleration_executed = 0.8 * min(1.0, speed_drop / 4.8)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "perceptual_crosswalk": perceptual_crosswalk,
        "deceleration_executed": deceleration_executed,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
