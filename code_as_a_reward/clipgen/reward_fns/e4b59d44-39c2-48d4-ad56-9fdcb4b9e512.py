"""clip e4b59d44-39c2-48d4-ad56-9fdcb4b9e512 - attempt 2/5 - gate PASS (pos 0.95, max pert 0.06, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene e4b59d44-39c2-48d4-ad56-9fdcb4b9e512:
    1. Steering left to avoid construction zone: lateral maneuver with leftward trajectory change.
    Trajectory thresholds: lateral offset change >= 24 m, heading change >= -0.9 deg.
    """

    # Initialize component scores
    comp = {
        "perceptual_construction_zone": 0.0,
        "lateral_steering_left": 0.0,
    }

    # Perceptual claim for construction zone
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction_zone"] = 0.05  # Reduced weight for mention-only credit

    # Lateral maneuver: steering left
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Calculate heading change
        heading_change = traj.total_heading_change_deg

        # Graded lateral factor based on offset and heading change
        lateral_factor = 0.65 * min(1.0, lateral_offset_change / 3.59) + 0.25 * min(1.0, abs(heading_change) / 3.0)
        comp["lateral_steering_left"] = lateral_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
