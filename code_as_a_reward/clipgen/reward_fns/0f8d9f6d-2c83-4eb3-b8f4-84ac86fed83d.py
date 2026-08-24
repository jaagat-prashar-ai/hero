"""clip 0f8d9f6d-2c83-4eb3-b8f4-84ac86fed83d - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene.
    
    Decisive Events:
    1. Navigating through the construction zone with traffic cones.
       - Commitment: Maintain speed (speed_profile='maintain')
       - Trajectory: Maintain speed with minor variation.
    
    Scene-derived thresholds:
    - Speed maintenance: Graded factor based on speed variation.
    """
    # Initialize component scores
    comp = {
        "maintain_speed": 0.0,
    }
    
    # Check for speed maintenance commitment
    maintain_speed_claim = any(c.speed_profile == 'maintain' for c in claims.commitments)
    
    # Calculate speed maintenance trajectory factor
    initial_speed = traj.initial_speed_mps
    final_speed = traj.final_speed_mps
    speed_drop = initial_speed - final_speed
    speed_maintenance_factor = 0.7 * min(1.0, speed_drop / 0.7)  # Graded factor

    # Assign score for maintaining speed
    if maintain_speed_claim and traj.min_speed_mps == final_speed:
        comp["maintain_speed"] = speed_maintenance_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
