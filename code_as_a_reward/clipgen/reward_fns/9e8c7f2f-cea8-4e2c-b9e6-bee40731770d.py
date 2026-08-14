"""clip 9e8c7f2f-cea8-4e2c-b9e6-bee40731770d - attempt 3/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on decisive events:
    1. Steering right to follow the temporary lane.
       - Perceptual mention: lane-related entities.
       - Commitment: Lateral maneuver (lane_change, nudge, merge, turn, enter, exit) excluding left.
       - Trajectory: Rightward heading change of at least -12 degrees.
    """

    # Initialize component scores
    perceptual_lane = 0.0
    lateral_maneuver = 0.0

    # Check for perceptual claims
    if any(p.entity in ('lane', 'construction_cones', 'barricades') for p in claims.perceptual):
        perceptual_lane = 0.1

    # Check for lateral commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Calculate heading change
        heading_change = traj.total_heading_change_deg
        if heading_change < 0:  # Rightward change
            lateral_maneuver = 0.9 * min(1.0, abs(heading_change) / 24.0)

    return {
        "perceptual_lane": perceptual_lane,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
