"""clip c45cac12-442b-4687-a251-7130af9c45b4 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.11, real rollout argmax 9)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the decisive event of yielding to a pedestrian.
    Scene-derived thresholds:
    - Yield to pedestrian: speed drop of at least 1.25 m/s, occurring primarily between t=0 and t=3.5 seconds.
    - Perceptual mention of pedestrian-related entities.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    yield_executed = 0.0

    # Check for perceptual mention of pedestrian-related entities
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for commitment to decelerate (yield) and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop, with a floor at half the GT magnitude
        yield_executed = 0.7 * min(1.0, speed_drop / 2.5)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "yield_executed": yield_executed,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
