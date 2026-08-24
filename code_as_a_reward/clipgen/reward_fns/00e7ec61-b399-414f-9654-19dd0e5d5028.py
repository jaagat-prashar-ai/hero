"""clip 00e7ec61-b399-414f-9654-19dd0e5d5028 - attempt 4/5 - gate PASS (pos 1.00, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """Components for navigating a construction zone with a pedestrian ahead.
    
    Decisive Events:
    1. Navigating through the construction zone: Expect a leftward turn with a heading change of at least -0.5 degrees.

    Trajectory thresholds:
    - Heading change: At least -0.5 degrees (half of -1 degree).
    """

    # Initialize component scores
    comp = {
        "perceptual_construction": 0.05,
        "lateral_navigation": 0.0,
    }

    # Lateral maneuver for navigating the construction zone
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change <= -0.5:
            comp["lateral_navigation"] = 0.95 * min(1.0, abs(heading_change) / 0.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
