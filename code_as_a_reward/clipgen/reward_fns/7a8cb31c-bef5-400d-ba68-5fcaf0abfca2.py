"""clip 7a8cb31c-bef5-400d-ba68-5fcaf0abfca2 - attempt 2/5 - gate PASS (pos 0.90, max pert 0.20, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on decisive events:
    1. Acceleration after traffic light turns green.
       - Perceptual mention of 'signal' or 'vehicle_generic'.
       - Commitment to 'accelerate'.
       - Trajectory shows a speed increase of at least 4.5 m/s.
    2. Maintaining safe distance from nearby obstacles.
       - Perceptual mention of 'pedestrian' or 'vehicle_generic'.
    """

    # Initialize component scores
    perceptual_signal = 0.0
    perceptual_obstacles = 0.0
    accelerate_commitment = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('signal', 'vehicle_generic') for p in claims.perceptual):
        perceptual_signal = 0.1

    if any(p.entity in ('pedestrian', 'vehicle_generic') for p in claims.perceptual):
        perceptual_obstacles = 0.1

    # Check for acceleration commitment
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Calculate speed increase
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        accelerate_commitment = 0.7 * min(1.0, speed_increase / 9.0)

    # Return component scores
    return {
        "perceptual_signal": perceptual_signal,
        "perceptual_obstacles": perceptual_obstacles,
        "accelerate_commitment": accelerate_commitment
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
