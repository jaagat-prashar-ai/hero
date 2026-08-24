"""clip 25159f91-8de0-4230-9f89-805f8703dd39 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.29, real rollout argmax 9)"""
def components(claims, traj):
    """Components for scene 25159f91-8de0-4230-9f89-805f8703dd39:
    - Lateral shift to the left for construction zone (graded on lateral offset)
    - Perception of construction zone or cones (mention-only, reduced weight)
    """
    # Initialize component scores
    comp = {
        "lateral_shift": 0.0,
        "perception_construction": 0.0,
    }

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the lateral offset change
        final_offset = traj.final_lateral_offset_m
        # Graded factor for lateral shift
        comp["lateral_shift"] = 0.70 * min(1.0, abs(final_offset) / 0.98)

    # Check for relevant perceptual claims
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perception_construction"] = 0.00  # Remove mention-only credit

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
