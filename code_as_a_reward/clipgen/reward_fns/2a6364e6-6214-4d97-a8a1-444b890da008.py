"""clip 2a6364e6-6214-4d97-a8a1-444b890da008 - attempt 1/5 - gate PASS (pos 0.75, max pert 0.25, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with lead vehicle and traffic light.
    
    Decisive Events:
    1. Lead Vehicle and Traffic Light: Expect deceleration due to the lead vehicle and traffic light.
       - Perceptual mention: {'lead_vehicle', 'signal'}
       - Commitment: speed_profile='decelerate'
       - Trajectory: Speed drop of at least 0.25 m/s, graded for more.
    2. Lateral Positioning: Maintain stable lateral position.
       - Perceptual mention: {'lane'}
       - No specific commitment required.
       - Trajectory: Lateral offset within 0.5 m of GT's final offset.
    """
    comp = {
        "mention_lead_vehicle": 0.0,
        "mention_signal": 0.0,
        "mention_lane": 0.0,
        "decelerate_execution": 0.0,
        "lateral_stability": 0.0,
    }

    # Perceptual mentions
    if any(p.entity in {'lead_vehicle'} for p in claims.perceptual):
        comp["mention_lead_vehicle"] = 0.05
    if any(p.entity in {'signal'} for p in claims.perceptual):
        comp["mention_signal"] = 0.05
    if any(p.entity in {'lane'} for p in claims.perceptual):
        comp["mention_lane"] = 0.05

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        comp["decelerate_execution"] = 0.5 * min(1.0, speed_drop / 0.5)

    # Lateral stability
    lateral_offset = abs(traj.final_lateral_offset_m)
    comp["lateral_stability"] = 0.2 * min(1.0, 0.92 / lateral_offset)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
