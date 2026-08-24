"""clip 28a29f95-3e75-4c98-9f08-b5c123a4428a - attempt 2/5 - gate PASS (pos 1.00, max pert 0.41, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with right turn and yielding to construction area.
    
    Decisive events:
    1. Right Turn Initiation: Expect a right turn maneuver with a significant heading change.
    2. Yielding to Construction Area: Expect deceleration in response to the construction area.
    
    Scene-derived thresholds:
    - Right turn: Heading change >= 47.75 degrees.
    - Yielding: Speed drop >= 0.75 m/s.
    """
    comp = {
        "mention_construction": 0.0,
        "right_turn_executed": 0.0,
        "yield_executed": 0.0
    }

    # Perceptual mention of construction area
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["mention_construction"] = 0.05

    # Right Turn Execution
    if any(c.maneuver == 'turn' and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change <= -47.75:  # Ensure right turn
            comp["right_turn_executed"] = 0.45 * min(1.0, abs(heading_change) / 75.0)

    # Yield Execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 0.75:
            comp["yield_executed"] = 0.50 * min(1.0, speed_drop / 2.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
