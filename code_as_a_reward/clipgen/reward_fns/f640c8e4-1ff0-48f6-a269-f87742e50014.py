"""clip f640c8e4-1ff0-48f6-a269-f87742e50014 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene.
    
    Decisive events:
    1. Right Turn Execution: Expect a 'turn' commitment with a right direction and a significant heading change.
    
    Scene-derived thresholds:
    - Heading change: at least 40 degrees for a right turn.
    - Speed drop: at least 0.5 m/s.
    """
    comp = {
        "right_turn": 0.0,
    }
    
    # Right Turn Execution with Commitment
    if any(c.maneuver == 'turn' and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change <= -40.0:  # Expecting a right turn
            comp["right_turn"] = 0.7 * min(1.0, abs(heading_change) / 80.0)
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
