"""clip 98b987b1-b555-415f-8229-9a17583be116 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scene with decisive event: Steering right to avoid construction zone.
    - Perceptual: Mention of construction-related entities.
    - Commitment: Lateral maneuver to the right (e.g., lane_change, nudge).
    - Trajectory: Rightward heading change of at least -3.0 degrees.
    """

    # Initialize component scores
    perceptual_credit = 0.0
    lateral_commitment_credit = 0.0

    # Check for perceptual mentions of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_credit = 0.05  # Reduced weight for mention

    # Check for lateral commitment to the right
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Graded trajectory factor for rightward heading change
        heading_change = traj.total_heading_change_deg
        if heading_change < 0:  # Rightward change is negative
            lateral_commitment_credit = 0.65 * min(1.0, abs(heading_change) / 7.0)  # Graded factor

    # Return component scores
    return {
        "perceptual_mention": perceptual_credit,
        "lateral_commitment_execution": lateral_commitment_credit,
    }

def reward(claims, traj):
    # Sum components and clamp to [0, 1]
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
