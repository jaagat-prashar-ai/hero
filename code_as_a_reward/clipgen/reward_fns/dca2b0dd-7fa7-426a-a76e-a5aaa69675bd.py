"""clip dca2b0dd-7fa7-426a-a76e-a5aaa69675bd - attempt 5/5 - gate PASS (pos 0.72, max pert 0.05, real rollout argmax 4)"""
def components(claims, traj):
    """
    Components for scoring the rollout's faithfulness to the scene:
    - Resumption of speed: acceleration commitment and trajectory speed increase.
    - Perceptual mentions: intersection.
    """
    comp = {
        "accelerate_commitment": 0.0,
        "speed_increase": 0.0,
        "mention_intersection": 0.0,
    }

    # Check for acceleration commitment and trajectory execution
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        speed_increase = final_speed - initial_speed
        if speed_increase > 3.15:  # Half of the GT speed increase
            comp["accelerate_commitment"] = 0.4
            comp["speed_increase"] = 0.5 * min(1.0, speed_increase / 6.3)

    # Perceptual mentions
    if any(p.entity == 'intersection' for p in claims.perceptual):
        comp["mention_intersection"] = 0.05

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
