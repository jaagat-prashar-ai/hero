"""clip d7f060c3-b9fe-4255-b548-8720f85c3534 - attempt 1/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 5)"""
def components(claims, traj):
    """Compute reward components for the scene: deceleration in response to traffic barriers and maintaining lane position.
    
    Decisive events:
    1. Deceleration in response to traffic barriers: Expect a speed drop of at least 4.25 m/s, with a commitment to decelerate.
    2. Maintaining lane position: Expect a lateral offset within |5.53 m|, but this is not tied to a specific maneuver claim.
    
    Scene-derived thresholds:
    - Speed drop floor: 4.25 m/s (half of GT's 8.5 m/s drop)
    - Lateral offset: |5.53 m| (not a decisive maneuver, so minimal weight)
    """
    # Initialize component scores
    comp = {
        "perceptual_barriers": 0.0,
        "perceptual_construction_zone": 0.0,
        "decelerate_executed": 0.0,
        "maintain_lane": 0.0
    }
    
    # Perceptual claims
    if any(p.entity in ('barricades', 'construction_cones') for p in claims.perceptual):
        comp["perceptual_barriers"] = 0.05
    if any(p.entity in ('work_zone', 'construction_cones') for p in claims.perceptual):
        comp["perceptual_construction_zone"] = 0.05
    
    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration
        comp["decelerate_executed"] = 0.7 * min(1.0, speed_drop / 8.5)
    
    # Maintaining lane position (minimal weight, not tied to a specific maneuver)
    lateral_offset = abs(traj.final_lateral_offset_m)
    if lateral_offset <= 5.53:
        comp["maintain_lane"] = 0.05
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
