"""clip 56ff5406-cc81-4477-b384-cbdaf87698ba - attempt 2/5 - gate PASS (pos 0.90, max pert 0.30, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Gentle acceleration through intersection while navigating around construction cones.
       - Perceptual mention of 'construction_cones' or 'intersection'.
       - Commitment to 'accelerate'.
       - Trajectory shows speed increase, gated by the commitment claim.
    2. Awareness of nearby vehicles.
       - Perceptual mention of 'vehicle_generic'.
    """
    # Initialize component scores
    perceptual_construction = 0.0
    perceptual_vehicle = 0.0
    accelerate_commitment = 0.0
    speed_increase_execution = 0.0

    # Check perceptual claims
    if any(p.entity in ('construction_cones', 'intersection') for p in claims.perceptual):
        perceptual_construction = 0.1

    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        perceptual_vehicle = 0.1

    # Check commitment claims
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        accelerate_commitment = 0.2
        # Check trajectory execution for acceleration, gated by commitment
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        if speed_increase > 0:
            speed_increase_execution = 0.6 * min(1.0, speed_increase / 2.0)

    return {
        "perceptual_construction": perceptual_construction,
        "perceptual_vehicle": perceptual_vehicle,
        "accelerate_commitment": accelerate_commitment,
        "speed_increase_execution": speed_increase_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
