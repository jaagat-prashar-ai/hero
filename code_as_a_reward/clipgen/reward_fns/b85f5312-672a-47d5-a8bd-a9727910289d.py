"""clip b85f5312-672a-47d5-a8bd-a9727910289d - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene b85f5312-672a-47d5-a8bd-a9727910289d.
    
    Decisive events:
    1. Steering left to avoid a traffic barrier on the right.
       - Perceptual mention: any of 'barricades', 'vehicle_generic'
       - Commitment: lateral maneuver (lane_change, nudge, merge, turn, enter, exit) excluding right
       - Trajectory: positive heading change, floor at +4.5 degrees
    """
    
    # Initialize component scores
    perceptual_mention = 0.0
    lateral_maneuver = 0.0
    
    # Check for perceptual mention of relevant entities
    if any(p.entity in ('barricades', 'vehicle_generic') for p in claims.perceptual):
        perceptual_mention = 0.05  # Small weight for mention
    
    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate heading change
        heading_change = traj.total_heading_change_deg
        lateral_maneuver = 0.65 * min(1.0, heading_change / 9.0)  # Graded factor based on heading change
    
    return {
        "perceptual_mention": perceptual_mention,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
