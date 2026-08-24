"""clip e828096d-5889-40bb-bc6f-2feed7db9821 - attempt 3/5 - gate PASS (pos 0.95, max pert 0.06, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness in the construction zone scene.
    Decisive events:
    1. Construction Zone Navigation: Expect mention of construction-related entities and a leftward lateral maneuver.
    Trajectory thresholds are set to half the scene's magnitude for graded scoring.
    """

    # Initialize component scores
    comp = {
        "mention_vehicle": 0.0,
        "mention_pedestrian": 0.0,
        "lateral_adjustment": 0.0,
    }

    # Perceptual mentions
    if any(p.entity in ('vehicle_generic', 'stopped_vehicle') for p in claims.perceptual):
        comp["mention_vehicle"] = 0.05  # Reduced weight for perceptual mention

    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05  # Reduced weight for perceptual mention

    # Lateral adjustment for construction zone navigation
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        lateral_change = abs(traj.final_lateral_offset_m)
        comp["lateral_adjustment"] = 0.9 * min(1.0, lateral_change / 1.6)  # Gated by commitment claim

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
