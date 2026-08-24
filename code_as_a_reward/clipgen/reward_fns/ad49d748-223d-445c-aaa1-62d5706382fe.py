"""clip ad49d748-223d-445c-aaa1-62d5706382fe - attempt 3/5 - gate PASS (pos 0.75, max pert 0.08, real rollout argmax 2)"""
def components(claims, traj):
    """Components for rewarding a rollout based on steering right to merge back into the lane.
    
    Decisive Event: Steering right to merge back into the lane after passing through a construction zone.
    - Perceptual mention: lane, work_zone, construction_cones, barricades
    - Commitment family: merge (rightward direction)
    - Trajectory expectations: Rightward heading change of at least -2.0 degrees and lateral offset of at least -1.6 meters.
    """
    perceptual_credit = 0.0
    commitment_credit = 0.0

    # Perceptual mention credit (mention-only)
    if any(p.entity in ('lane', 'work_zone', 'construction_cones', 'barricades') for p in claims.perceptual):
        perceptual_credit = 0.05  # Reduced weight to allow more for commitment

    # Commitment credit for merging right
    if any(c.maneuver in ('merge', 'lane_change', 'nudge', 'turn', 'enter', 'exit') and (c.direction is None or c.direction != 'left') for c in claims.commitments):
        # Trajectory analysis for rightward merge
        lateral_offset = traj.final_lateral_offset_m

        # Graded trajectory factor for lateral offset
        lateral_offset_factor = 0.7 * min(1.0, abs(lateral_offset) / 2.0) if lateral_offset < 0 else 0.0

        # Total commitment credit
        commitment_credit = lateral_offset_factor

    return {
        "perceptual_mention": perceptual_credit,
        "commitment_execution": commitment_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
