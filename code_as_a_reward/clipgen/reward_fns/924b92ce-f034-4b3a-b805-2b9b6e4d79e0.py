"""clip 924b92ce-f034-4b3a-b805-2b9b6e4d79e0 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring a rollout's faithfulness to the scene:
    - Construction Zone Awareness: Perceptual mention of construction-related entities.
    - Rightward Steering Execution: Lateral maneuver to the right with graded heading change.
    - Pedestrian Awareness: Perceptual mention of pedestrians.
    
    Scene-derived thresholds:
    - Rightward heading change of at least -2.5 degrees.
    - No significant speed drop required for pedestrians.
    """
    # Initialize component scores
    comp = {
        "construction_zone_awareness": 0.0,
        "rightward_steering_execution": 0.0,
        "pedestrian_awareness": 0.0,
    }

    # Construction Zone Awareness
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["construction_zone_awareness"] = 0.1

    # Rightward Steering Execution
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change < 0:  # Rightward change
            comp["rightward_steering_execution"] = 0.6 * min(1.0, abs(heading_change) / 5.0)

    # Pedestrian Awareness
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["pedestrian_awareness"] = 0.1

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
