"""clip 7c5a1404-96e8-4c56-b9e0-d7e2a121eaf1 - attempt 2/5 - gate PASS (pos 0.76, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive events:
    1. Steering right through the construction zone.
       - Perceptual mention of construction-related entities.
       - Commitment to a rightward lateral maneuver.
       - Trajectory showing a rightward heading change.
    
    Trajectory thresholds are set to approximately half the ground truth's
    magnitudes to allow for variability in execution.
    """
    # Initialize component scores
    comp = {
        "perceptual_construction": 0.0,
        "lateral_maneuver": 0.0,
        "heading_change": 0.0
    }

    # (a) Perceptual mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.1

    # (b) Commitment to a rightward lateral maneuver
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # (c) Trajectory showing a rightward heading change
        heading_change = traj.total_heading_change_deg
        if heading_change < 0:  # Ensure rightward turn
            comp["lateral_maneuver"] = 0.2
            comp["heading_change"] = 0.5 * min(1.0, abs(heading_change) / 6.7)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
