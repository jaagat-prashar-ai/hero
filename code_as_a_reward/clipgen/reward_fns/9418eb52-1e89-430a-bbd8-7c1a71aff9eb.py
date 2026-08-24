"""clip 9418eb52-1e89-430a-bbd8-7c1a71aff9eb - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with right turn:
    - Right Turn: heading change >= -5.0 degrees, commitment 'turn' right
    """
    comp = {
        "right_turn_commitment": 0.0,
    }

    # Check for right turn commitment
    if any(c.maneuver == 'turn' and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change <= -5.0:
            comp["right_turn_commitment"] = 0.7 * min(1.0, abs(heading_change) / 10.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
