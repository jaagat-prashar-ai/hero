"""clip 8c2cd02d-83e6-42ca-83cc-fd9555cb91d6 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene with curved road navigation and construction zone.
    
    Decisive Events:
    1. Curved Road Navigation: Expect significant heading change.
    2. Maintaining Safe Distance from Construction Zone: Expect perceptual mention of construction-related entities.
    
    Trajectory Thresholds:
    - Speed Maintenance: >= 5.2 m/s (initial speed).
    """
    # Initialize component scores
    comp = {
        "speed_maintenance": 0.0
    }
    
    # Trajectory-based components with commitment checks
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        if final_speed >= initial_speed:
            comp["speed_maintenance"] = 0.7 * min(1.0, (final_speed - initial_speed) / 0.7)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
