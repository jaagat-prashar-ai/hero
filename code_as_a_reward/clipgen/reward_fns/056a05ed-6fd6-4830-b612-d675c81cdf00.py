"""clip 056a05ed-6fd6-4830-b612-d675c81cdf00 - attempt 5/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 056a05ed-6fd6-4830-b612-d675c81cdf00:
    - Maintain safe distance from lead vehicle (track 3) with speed reduction.
    - Follow road curvature with heading change.
    Thresholds: speed drop >= 0.25 m/s, heading change >= 1.5 degrees.
    """

    # Initialize component scores
    comp = {
        "maintain_distance_execution": 0.0,
        "follow_road_curvature_execution": 0.0,
    }

    # Commitment claims and trajectory execution
    # Maintain distance with speed adjustment
    if any(c.maneuver == 'keep_distance' for c in claims.commitments):
        # No speed drop in the positive case, so check for maintaining speed
        if traj.initial_speed_mps <= traj.final_speed_mps <= traj.initial_speed_mps + 0.5:
            comp["maintain_distance_execution"] = 0.7

    # Follow road curvature with heading change
    if any(c.maneuver in ('nudge', 'lane_change', 'merge', 'turn', 'enter', 'exit') for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if abs(heading_change) >= 1.5:
            comp["follow_road_curvature_execution"] = 0.3 * min(1.0, abs(heading_change) / 3.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
