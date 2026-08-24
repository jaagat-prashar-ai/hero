"""clip 16638d32-3382-4f83-be1a-4ebd8f893767 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 8)"""
def components(claims, traj):
    """Components for scene 16638d32-3382-4f83-be1a-4ebd8f893767:
    - Accelerate following the lead vehicle while navigating through the construction zone.
    - Thresholds: speed increase >= 3.0 m/s.
    """

    # Initialize component scores
    perceptual_work_zone = 0.0
    perceptual_lead_vehicle = 0.0
    accelerate_execution = 0.0

    # Perceptual claims
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_work_zone = 0.1

    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_lead_vehicle = 0.1

    # Commitment claims and trajectory checks
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Speed increase check
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        accelerate_execution = 0.8 * min(1.0, speed_increase / 6.0)

    return {
        "perceptual_work_zone": perceptual_work_zone,
        "perceptual_lead_vehicle": perceptual_lead_vehicle,
        "accelerate_execution": accelerate_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
