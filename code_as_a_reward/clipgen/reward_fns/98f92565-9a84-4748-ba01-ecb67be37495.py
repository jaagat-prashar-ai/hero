"""clip 98f92565-9a84-4748-ba01-ecb67be37495 - attempt 4/5 - gate PASS (pos 0.75, max pert 0.07, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene 98f92565-9a84-4748-ba01-ecb67be37495:
    - Steering right through a construction zone (lateral maneuver)
    - Presence of nearby pedestrians (perceptual mention)
    Trajectory expectations: rightward lateral offset change of at least +4.31 m.
    """

    # Initialize component scores
    comp = {
        "steer_right": 0.0,
        "mention_construction": 0.0,
        "mention_pedestrian": 0.0
    }

    # Check for perceptual mentions
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["mention_construction"] = 0.05  # Reduced weight to allow more for commitment

    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05  # Reduced weight to allow more for commitment

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Calculate the lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor for lateral offset change
        if lateral_offset_change < 0:  # Ensure rightward movement
            comp["steer_right"] = 0.7 * min(1.0, abs(lateral_offset_change) / 8.62)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
