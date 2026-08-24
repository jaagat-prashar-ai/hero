"""clip 738f32fd-74ac-4baf-8cd0-61aec59b8aae - attempt 1/5 - gate PASS (pos 0.90, max pert 0.30, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with a lead vehicle requiring a stop or deceleration.
    
    Decisive Event: Deceleration and Stop Behind Lead Vehicle
    - Perceptual: Mention of 'lead_vehicle' or 'vehicle_generic'
    - Commitment: 'decelerate' family (stop/yield/wait/decelerate)
    - Trajectory: Maintain a low speed, consistent with stopping or creeping
    
    Trajectory thresholds:
    - Speed drop: Graded factor based on maintaining low speed
    - Lateral offset: Minimal change, indicating staying in lane
    """
    comp = {}

    # Perceptual component: Mention of lead vehicle or generic vehicle
    comp['perceptual_vehicle'] = 0.1 * any(
        p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual
    )

    # Commitment component: Decelerate family
    decelerate_commitment = any(
        c.speed_profile == 'decelerate' for c in claims.commitments
    )

    # Trajectory component: Speed drop and low speed maintenance
    if traj.n_waypoints > 0:
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        speed_drop = initial_speed - final_speed
        comp['decelerate_execution'] = 0.6 * min(1.0, (initial_speed - final_speed) / 1.0) if decelerate_commitment else 0.0

    # Lateral offset component: Minimal lateral movement
    lateral_offset_change = abs(traj.final_lateral_offset_m)
    comp['lateral_stability'] = 0.2 * (1.0 if lateral_offset_change < 0.1 else 0.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
