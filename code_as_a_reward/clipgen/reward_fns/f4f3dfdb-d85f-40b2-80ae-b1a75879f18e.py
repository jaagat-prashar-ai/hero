"""clip f4f3dfdb-d85f-40b2-80ae-b1a75879f18e - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 3)"""
def components(claims, traj):
    """Components for the scene where the expert steers left to avoid construction equipment and traffic cones.
    
    Decisive Event: Steering left to avoid obstacles on the right side of the road.
    - Perceptual mention of construction-related entities.
    - Lateral maneuver commitment to steer left (lane_change, nudge, turn).
    - Trajectory should show a significant leftward heading change, at least +2 degrees.
    """

    # Initialize component scores
    perceptual_mention = 0.05  # Small mention-only credit
    lateral_maneuver = 0.0

    # Check for perceptual mentions of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_mention = 0.05  # Small additive weight for mention

    # Check for lateral maneuver commitment and trajectory execution
    if any(c.maneuver in ('lane_change', 'nudge', 'turn') and c.direction != 'right' for c in claims.commitments):
        # Calculate the heading change over the trajectory
        heading_change = traj.total_heading_change_deg
        # Graded factor for heading change, with a floor at half the GT magnitude
        if heading_change >= 2.0:  # Minimum meaningful heading change
            lateral_maneuver = 0.65 * min(1.0, heading_change / 4.0)

    return {
        "perceptual_mention": perceptual_mention,
        "lateral_maneuver": lateral_maneuver,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
