"""clip 63af8530-b198-4258-9eaf-4bbc4fb63046 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.12, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scene 63af8530-b198-4258-9eaf-4bbc4fb63046:
    - Lateral shift to the left to avoid obstacles on the right.
    - Perceptual mention of obstacles like parked vehicles.
    - Trajectory should show a leftward shift with a minimum lateral offset of +2.73 m and a heading change of at least +5 degrees.
    """

    # Initialize component scores
    perceptual_mention = 0.0
    lateral_maneuver = 0.0

    # Check for perceptual mention of relevant obstacles
    if any(p.entity in ('vehicle_generic', 'stopped_vehicle') for p in claims.perceptual):
        perceptual_mention = 0.1

    # Check for lateral maneuver commitment and corresponding trajectory execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the lateral offset achieved
        final_offset = traj.final_lateral_offset_m
        lateral_factor = 0.45 * min(1.0, final_offset / 5.46)  # Graded factor based on final offset
        # Calculate the heading change achieved
        heading_change = traj.total_heading_change_deg
        heading_factor = 0.45 * min(1.0, heading_change / 9.9)  # Graded factor based on heading change
        # Combine the factors
        lateral_maneuver = lateral_factor + heading_factor

    return {
        "perceptual_mention": perceptual_mention,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
