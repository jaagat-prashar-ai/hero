"""clip 5a91124d-e18c-4650-85dc-a2b3b6cf0f48 - attempt 3/5 - gate PASS (pos 0.90, max pert 0.00, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene with pedestrians and road curvature:
    - Deceleration in response to pedestrians on the left.
    - Rightward steering to follow road curvature.
    - Thresholds: heading change >= 6.5 degrees.
    """
    comp = {
        "perceptual_pedestrian": 0.0,
        "lateral_steer_right": 0.0
    }

    # Perceptual mention of pedestrians
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Lateral maneuver commitment and trajectory check
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change <= -6.5:
            comp["lateral_steer_right"] = 0.9 * min(1.0, abs(heading_change) / 13.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
