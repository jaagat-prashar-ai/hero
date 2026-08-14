"""clip a34c532f-d452-4678-9111-ac56480972ed - attempt 2/5 - gate PASS (pos 0.90, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """Components for the scene where the ego vehicle resumes speed after a traffic light turns green. The trajectory should show acceleration."""
    
    # Initialize component scores
    components = {
        "mention_signal": 0.0,
        "mention_construction_zone": 0.0,
        "accelerate_executed": 0.0
    }

    # Perceptual mentions
    if any(p.entity in {"signal"} for p in claims.perceptual):
        components["mention_signal"] = 0.1

    if any(p.entity in {"work_zone", "construction_cones", "barricades"} for p in claims.perceptual):
        components["mention_construction_zone"] = 0.1

    # Commitment and trajectory checks
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Calculate speed increase
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        # Graded factor for acceleration
        components["accelerate_executed"] = 0.8 * min(1.0, speed_increase / 6.0)

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
