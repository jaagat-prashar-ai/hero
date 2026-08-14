"""clip a987d18a-92e2-4c13-b61e-fda0de5e685d - attempt 1/5 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive event:
    - Steering left to maintain a safe distance from the construction zone.
    - Trajectory expectations: leftward heading change of at least 3.1 degrees.
    - Perceptual mentions: work_zone, construction_cones, barricades.
    """

    # Initialize component scores
    perceptual_score = 0.0
    lateral_commitment_score = 0.0
    lateral_execution_score = 0.0

    # Check for relevant perceptual mentions
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades') for p in claims.perceptual):
        perceptual_score = 0.1  # Small additive weight for perceptual mention

    # Check for lateral commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right'
           for c in claims.commitments):
        lateral_commitment_score = 0.3  # Commitment credit for lateral maneuver

        # Calculate the heading change over the trajectory
        heading_change = traj.total_heading_change_deg

        # Graded trajectory factor for leftward heading change
        if heading_change < 0:  # Ensure leftward change
            lateral_execution_score = 0.6 * min(1.0, abs(heading_change) / 6.2)

    return {
        "perceptual_mention": perceptual_score,
        "lateral_commitment": lateral_commitment_score,
        "lateral_execution": lateral_execution_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
