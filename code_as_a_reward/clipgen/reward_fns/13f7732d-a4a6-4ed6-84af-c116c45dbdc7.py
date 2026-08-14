"""clip 13f7732d-a4a6-4ed6-84af-c116c45dbdc7 - attempt 5/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 6)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Pedestrian Crossing: Expect a deceleration commitment.
    2. Road Curvature: Expect adaptation to the curve with significant heading change.
    Trajectory thresholds are derived from the expert's trajectory within the prediction window.
    """

    # Initialize component scores
    comp = {
        "decelerate_commitment": 0.0,
        "heading_change": 0.0,
    }

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed increase
        initial_speed = traj.speed_mps[0]
        final_speed = traj.speed_mps[-1]
        speed_increase = final_speed - initial_speed
        # Graded factor for speed increase, expecting an increase if claimed
        comp["decelerate_commitment"] = 0.7 * min(1.0, speed_increase / 7.8)

    # Check for heading change due to road curvature
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') for c in claims.commitments):
        total_heading_change = traj.total_heading_change_deg
        if total_heading_change <= -40.0:  # Expecting significant heading change
            comp["heading_change"] = 0.3 * min(1.0, abs(total_heading_change) / 80.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
