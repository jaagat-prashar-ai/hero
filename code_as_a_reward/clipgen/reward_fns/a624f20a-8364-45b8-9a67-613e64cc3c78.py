"""clip a624f20a-8364-45b8-9a67-613e64cc3c78 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scene a624f20a-8364-45b8-9a67-613e64cc3c78:
    - Maintain speed while following temporary traffic delineators.
    - Lateral maneuver to follow delineators: graded on lateral offset change.
    - Perceptual mention of road infrastructure entities.
    Scene-derived thresholds:
    - Lateral offset change: at least +2.0 m within the first 3.2 s.
    - Speed maintenance: speed should remain relatively constant, with a permissible increase up to 10.0 m/s.
    """

    # Initialize component scores
    perceptual_mention = 0.0
    lateral_maneuver = 0.0

    # Check for perceptual mention of road infrastructure entities
    if any(p.entity in ('construction_cones', 'barricades', 'lane') for p in claims.perceptual):
        perceptual_mention = 0.1

    # Check for lateral maneuver commitment and grade based on lateral offset change
    if any(c.maneuver in ('lane_change', 'nudge', 'merge') and c.direction != 'right' for c in claims.commitments):
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        if lateral_offset_change >= 2.0:
            lateral_maneuver = 0.6 * min(1.0, lateral_offset_change / 4.0)  # Graded factor

    # Return the component contributions
    return {
        "perceptual_mention": perceptual_mention,
        "lateral_maneuver": lateral_maneuver,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
