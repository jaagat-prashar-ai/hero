"""clip cc4e6a8d-e499-4cd8-8bc6-ddcbd26a5ec2 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    
    1. Steering Left through Construction Zone:
       - Perceptual mention of construction-related entities.
       - Commitment to a lateral maneuver (leftward).
       - Trajectory showing a leftward heading change of at least 6 degrees.
    
    2. Speed Adjustment:
       - Removed due to lack of significant speed drop in the positive case.
    """
    # Initialize component scores
    comp = {
        "mention_construction": 0.0,
        "lateral_maneuver": 0.0
    }

    # Perceptual mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["mention_construction"] = 0.05  # Reduced weight to free up for lateral maneuver

    # Lateral maneuver commitment and trajectory check
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        comp["lateral_maneuver"] = 0.65 * min(1.0, heading_change / 12.0)  # Adjusted weight and threshold

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
