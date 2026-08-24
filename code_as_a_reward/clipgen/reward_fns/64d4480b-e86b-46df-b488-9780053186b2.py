"""clip 64d4480b-e86b-46df-b488-9780053186b2 - attempt 4/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with curved road navigation and speed adjustment.
    
    Decisive Events:
    1. Curved Road Navigation: Steer right through the construction zone.
       - Perceptual mention: construction_cones, work_zone
       - Commitment: lateral maneuver (lane_change, nudge, turn) excluding left
       - Trajectory: Heading change >= 47 degrees
    
    2. Speed Adjustment: Slow down in response to construction zone.
       - Perceptual mention: construction_cones, work_zone
       - Commitment: speed_profile='decelerate'
       - Trajectory: Speed drop >= 0.05 m/s
    """
    perceptual_weight = 0.1
    lateral_weight = 0.7
    slowing_weight = 0.2

    # Perceptual mentions
    saw_construction = any(p.entity in ('construction_cones', 'work_zone', 'barricades') for p in claims.perceptual)

    # Lateral maneuver commitment
    lateral_commitment = any(c.maneuver in ('lane_change', 'nudge', 'turn') and c.direction != 'left' for c in claims.commitments)
    heading_change = traj.total_heading_change_deg
    lateral_factor = 0.0
    if lateral_commitment and heading_change < 0:
        lateral_factor = lateral_weight * min(1.0, max(0.0, (abs(heading_change) - 47) / 47))

    # Speed adjustment commitment
    slowing_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    slowing_factor = 0.0
    if slowing_commitment:
        slowing_factor = slowing_weight * min(1.0, max(0.0, speed_drop / 0.05))

    return {
        "saw_construction": perceptual_weight if saw_construction else 0.0,
        "lateral_executed": lateral_factor,
        "slowing_executed": slowing_factor,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
