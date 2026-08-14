"""clip 1d06bd41-bd6e-40f9-870f-ab548cd5b8d2 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.05, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scoring the rollout based on the decisive event of approaching a construction zone.
    
    Decisive Event: Approach to Construction Zone
    - Perceptual mention of construction-related entities.
    - Commitment to nudge left.
    - Trajectory should show a slight leftward lateral adjustment.
    """
    # Initialize component scores
    perceptual_score = 0.0
    lateral_adjustment_score = 0.0

    # Check for perceptual mentions related to construction
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_score = 0.05  # Small additive weight for perceptual mention

    # Check for lateral adjustment commitment and corresponding trajectory execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = abs(traj.final_lateral_offset_m)
        # Graded score based on lateral offset change, with a floor at 0.2 m
        if lateral_offset_change >= 0.2:
            lateral_adjustment_score = 0.95 * min(1.0, lateral_offset_change / 0.64)

    # Return component scores
    return {
        "perceptual_mention": perceptual_score,
        "lateral_adjustment_execution": lateral_adjustment_score
    }

def reward(claims, traj):
    # Calculate the total score by summing the component scores and clamping between 0 and 1
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
