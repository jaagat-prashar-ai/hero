"""clip 3bb83a91-d6bf-4ce8-8a86-7545081daf9b - attempt 1/5 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of steering left to pass a construction vehicle.
    - Perceptual mention of construction-related entities.
    - Commitment to a lateral maneuver in the left direction.
    - Trajectory showing a leftward heading change.
    """

    # Initialize component scores
    perceptual_score = 0.0
    lateral_commitment_score = 0.0
    lateral_execution_score = 0.0

    # Check for perceptual mentions of construction-related entities
    if any(p.entity in ('vehicle_generic', 'work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_score = 0.1

    # Check for lateral maneuver commitment in the left direction
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        lateral_commitment_score = 0.3

        # Calculate the trajectory's leftward heading change
        heading_change = traj.total_heading_change_deg
        if heading_change < 0:  # Ensure it's a leftward change
            lateral_execution_score = 0.6 * min(1.0, abs(heading_change) / 15.0)  # Graded factor based on GT heading change

    return {
        "perceptual_mention": perceptual_score,
        "lateral_commitment": lateral_commitment_score,
        "lateral_execution": lateral_execution_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
