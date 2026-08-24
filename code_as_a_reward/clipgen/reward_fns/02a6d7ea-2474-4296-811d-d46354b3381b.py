"""clip 02a6d7ea-2474-4296-811d-d46354b3381b - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with gentle deceleration to maintain safe distance from pedestrians and cyclists.
    Decisive event: gentle deceleration for pedestrians and cyclists.
    Scene-derived thresholds: speed drop >= 1.3 m/s, graded from 0.0 to 2.6 m/s.
    """

    # Initialize component scores
    perceptual_score = 0.0
    deceleration_score = 0.0

    # Check for perceptual mentions of pedestrians or cyclists
    if any(p.entity in ('pedestrian', 'cyclist') for p in claims.perceptual):
        perceptual_score = 0.1  # Small additive weight for perceptual mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded deceleration score based on speed drop
        deceleration_score = 0.6 * min(1.0, speed_drop / 2.6)

    return {
        "perceptual_mention": perceptual_score,
        "deceleration_executed": deceleration_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
