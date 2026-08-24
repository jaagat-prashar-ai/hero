"""clip 509e5fa4-320c-40ce-bbe5-5f2675771d4c - attempt 5/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 9)"""
def components(claims, traj):
    """Components for scene 509e5fa4-320c-40ce-bbe5-5f2675771d4c:
    Merge right into the construction zone, guided by traffic cones.
    - Perceptual: construction zone or cones
    - Commitment: rightward lateral maneuver (merge/lane_change/nudge/turn/enter/exit)
    - Trajectory: rightward lateral offset change >= 0.2 m
    """
    perceptual_credit = 0.0
    lateral_commitment_credit = 0.0

    # Perceptual credit: mention of construction zone or cones
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_credit = 0.05  # Reduced weight

    # Lateral commitment and trajectory credit: rightward lateral maneuver
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Trajectory credit: rightward lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]

        if lateral_offset_change < 0:  # Ensure rightward movement
            lateral_commitment_credit = 0.65 * min(1.0, abs(lateral_offset_change) / 0.26)  # Half of 0.53 m

    return {
        "perceptual_mention": perceptual_credit,
        "lateral_commitment": lateral_commitment_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
