"""clip 3f631db8-8c57-4d19-a245-78f6098135a5 - attempt 2/5 - gate PASS (pos 0.93, max pert 0.17, real rollout argmax 10)"""
def components(claims, traj):
    """Components for reward function based on decisive events:
    - Steering left through a construction zone, following temporary lane markings.
    - Trajectory expectations: heading change of at least +4.5 degrees and lateral offset increase of at least +4.1 meters.
    """
    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Check for relevant perceptual mentions
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'lane') for p in claims.perceptual):
        perceptual_score = 0.1  # Small additive weight for perceptual mention

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate trajectory-based lateral factor
        heading_change = traj.total_heading_change_deg
        lateral_offset_increase = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        
        # Graded factor for heading change
        heading_factor = 0.5 * min(1.0, heading_change / 4.5)  # Adjusted to half the GT heading change
        # Graded factor for lateral offset increase
        lateral_offset_factor = 0.5 * min(1.0, lateral_offset_increase / 4.1)  # Adjusted to half the GT lateral offset increase
        
        # Combine factors for lateral maneuver score
        lateral_maneuver_score = heading_factor + lateral_offset_factor

    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver_executed": lateral_maneuver_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
