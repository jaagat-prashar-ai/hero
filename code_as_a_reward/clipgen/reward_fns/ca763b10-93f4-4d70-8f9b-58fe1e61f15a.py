"""clip ca763b10-93f4-4d70-8f9b-58fe1e61f15a - attempt 2/5 - gate PASS (pos 0.90, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene ca763b10-93f4-4d70-8f9b-58fe1e61f15a:
    - Steering left to avoid construction zone
    - Perceptual mention of construction-related entities
    - Lateral maneuver execution with significant leftward heading change
    """

    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Check for perceptual mentions of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_score = 0.1

    # Check for lateral maneuver commitment and corresponding trajectory execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the heading change over the trajectory
        heading_change = traj.total_heading_change_deg
        # Adjusted graded score based on heading change, with a floor at approximately half the measured magnitude
        lateral_maneuver_score = 0.9 * min(1.0, max(0.0, -heading_change / 2.0))

    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver_executed": lateral_maneuver_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
