"""clip 590477e5-8e96-4000-8f01-dc1d1a6f0e29 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scene 590477e5-8e96-4000-8f01-dc1d1a6f0e29:
    - Bus on the right: lateral nudge left to maintain safe distance.
    - Perceptual credit for mentioning bus or related entities.
    - Trajectory expectations: lateral offset change to the left, minimum 0.32 m.
    """

    # Initialize component scores
    perceptual_bus = 0.0
    lateral_nudge_left = 0.0

    # Check for perceptual mentions of the bus or related entities
    if any(p.entity in ('vehicle_generic', 'stopped_vehicle') for p in claims.perceptual):
        perceptual_bus = 0.1

    # Check for lateral nudge commitment to the left
    if any(c.maneuver in ('nudge', 'lane_change', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate the lateral offset change to the left
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        if lateral_offset_change > 0:  # Ensure it's a leftward change
            lateral_nudge_left = 0.6 * min(1.0, lateral_offset_change / 0.64)

    return {
        "perceptual_bus": perceptual_bus,
        "lateral_nudge_left": lateral_nudge_left,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
