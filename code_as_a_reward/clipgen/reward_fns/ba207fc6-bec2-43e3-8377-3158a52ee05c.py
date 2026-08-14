"""clip ba207fc6-bec2-43e3-8377-3158a52ee05c - attempt 2/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring a rollout based on steering left to follow a temporary lane delineated by traffic cones.
    
    Decisive Events:
    1. Speed Reduction: Expect a speed drop with a commitment to decelerate.
       - Trajectory: Expect a speed drop of at least 1.8 m/s (half of GT's 3.6 m/s).
    2. Perceptual Mention: Expect mention of lane boundaries or construction cones.
    
    Scene-derived thresholds:
    - Speed drop: Minimum 1.8 m/s (half of GT's 3.6 m/s).
    """
    perceptual_weight = 0.1
    speed_reduction_weight = 0.9

    # Perceptual mention of lane boundaries or construction cones
    saw_lane_or_cones = any(p.entity in ('lane', 'construction_cones') for p in claims.perceptual)
    perceptual_score = perceptual_weight if saw_lane_or_cones else 0.0

    # Speed reduction commitment
    speed_reduction_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Trajectory analysis for speed reduction
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps

    # Graded factor for speed drop
    speed_reduction_score = 0.0
    if speed_reduction_commitment:
        speed_reduction_score = speed_reduction_weight * min(1.0, speed_drop / 3.6)

    return {
        "perceptual_mention": perceptual_score,
        "speed_reduction": speed_reduction_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
