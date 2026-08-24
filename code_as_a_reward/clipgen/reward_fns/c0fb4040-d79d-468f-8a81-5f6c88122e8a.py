"""clip c0fb4040-d79d-468f-8a81-5f6c88122e8a - attempt 1/5 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the decisive event:
    - Steering left to avoid a parked vehicle in the same lane.
    - Perceptual mention of 'vehicle_generic' or 'lane'.
    - Lateral maneuver commitment (lane_change/nudge) excluding right direction.
    - Trajectory showing a leftward lateral offset change.
    """

    # Initialize component scores
    perceptual_score = 0.0
    lateral_commitment_score = 0.0
    lateral_execution_score = 0.0

    # Check for perceptual mentions of relevant entities
    if any(p.entity in ('vehicle_generic', 'lane') for p in claims.perceptual):
        perceptual_score = 0.1  # Small additive weight for mention

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor for lateral execution
        lateral_execution_score = 0.5 * min(1.0, lateral_offset_change / 0.50)
        lateral_commitment_score = 0.4  # Weight for commitment presence

    # Return component scores
    return {
        "perceptual_mention": perceptual_score,
        "lateral_commitment": lateral_commitment_score,
        "lateral_execution": lateral_execution_score
    }

def reward(claims, traj):
    # Calculate total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
