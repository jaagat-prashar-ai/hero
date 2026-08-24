"""clip 3000377d-e766-4b9e-b266-a6c5d70dd2f1 - attempt 3/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 8)"""
def components(claims, traj):
    """
    Components for scene 3000377d-e766-4b9e-b266-a6c5d70dd2f1:
    - Deceleration for pedestrian: speed drop >= 3.65 m/s, graded.
    - Perceptual mention of pedestrian: small additive credit.
    - Road curvature is secondary and not separately scored.
    """
    comp = {
        "decelerate_for_pedestrian": 0.0,
        "mention_pedestrian": 0.0,
    }

    # Check for perceptual mention of pedestrian
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded factor for speed drop, floor at half the GT drop
        if speed_drop >= 3.65:
            comp["decelerate_for_pedestrian"] = 0.7 * min(1.0, speed_drop / 7.3)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
