"""clip 7f0d9779-fb49-4607-b82b-93f9c798d2d0 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring a rollout's faithfulness in resuming speed after passing a construction zone.
    
    Decisive Event: Gentle acceleration to resume speed after passing the construction zone.
    Scene-derived thresholds:
    - Perceptual mention of 'work_zone', 'construction_cones', or 'barricades'.
    - Commitment to 'accelerate' (speed_profile='accelerate').
    - Trajectory should show a speed increase of at least 0.8 m/s over the window.
    """
    perceptual_credit = 0.1 if any(p.entity in ('work_zone', 'construction_cones', 'barricades') for p in claims.perceptual) else 0.0
    
    # Check for acceleration commitment
    commitment_accelerate = any(c.speed_profile == 'accelerate' for c in claims.commitments)
    
    # Calculate speed increase
    speed_increase = traj.final_speed_mps - traj.initial_speed_mps
    graded_speeding = 0.7 * min(1.0, speed_increase / 0.8) if commitment_accelerate else 0.0
    
    return {
        "perceptual_mention": perceptual_credit,
        "accelerate_executed": graded_speeding,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
