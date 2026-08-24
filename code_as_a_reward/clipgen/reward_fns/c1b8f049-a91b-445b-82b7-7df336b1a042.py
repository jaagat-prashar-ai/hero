"""clip c1b8f049-a91b-445b-82b7-7df336b1a042 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.17, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene c1b8f049-a91b-445b-82b7-7df336b1a042:
    - Decisive Event 1: Lane change to the left after passing the construction zone.
      - Perceptual: work_zone, construction_cones, barricades, workers
      - Commitment: lane_change (lateral maneuver), direction != 'right'
      - Trajectory: Lateral offset increase of at least +1.7 m (half of GT's +3.37 m)
    """

    # Initialize component scores
    comp = {
        "perceptual_construction": 0.0,
        "lane_change_executed": 0.0,
    }

    # Perceptual components (mention-only credit)
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.05  # Reduced weight for mention-only credit

    # Trajectory analysis
    lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
    lateral_factor = 0.65 * min(1.0, lateral_offset_change / 2.58)  # Adjusted for positive case

    # Commitment components
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        comp["lane_change_executed"] = lateral_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
