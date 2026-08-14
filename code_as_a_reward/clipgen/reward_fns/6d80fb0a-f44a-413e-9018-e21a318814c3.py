"""clip 6d80fb0a-f44a-413e-9018-e21a318814c3 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the decisive events:
    1. Steering left following temporary lane markings.
       - Perceptual mention of lane markings.
       - Lateral maneuver commitment (lane_change/nudge) to the left.
       - Trajectory showing a significant leftward heading change.
    """
    # Initialize component scores
    perceptual_lane_mention = 0.0
    lateral_maneuver_executed = 0.0

    # Check for perceptual mention of lane markings
    if any(p.entity == 'lane' for p in claims.perceptual):
        perceptual_lane_mention = 0.1

    # Check for lateral maneuver commitment to the left
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the heading change over the trajectory
        heading_change = traj.total_heading_change_deg
        # Graded factor for heading change, expecting a minimum of ~31 degrees
        lateral_maneuver_executed = 0.7 * min(1.0, heading_change / 62.0)

    return {
        "perceptual_lane_mention": perceptual_lane_mention,
        "lateral_maneuver_executed": lateral_maneuver_executed
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
