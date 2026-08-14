"""clip 0bb8b5ba-be70-4ac8-8e51-fe1cc8b5c3b6 - attempt 3/5 - gate PASS (pos 0.95, max pert 0.52, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive events:
    - Cyclists on the right prompt a slight leftward steering adjustment.
    - Perceptual mention of cyclists.
    - Lateral maneuver to nudge left.
    - Graded lateral offset change for trajectory execution.
    """

    # Initialize component scores
    perceptual_cyclist = 0.0
    lateral_nudge = 0.0
    lateral_execution = 0.0

    # Check for perceptual mention of cyclists
    if any(p.entity in ('cyclist',) for p in claims.perceptual):
        perceptual_cyclist = 0.05  # Mention-only credit

    # Check for lateral maneuver commitment to nudge left
    if any(c.maneuver in ('nudge', 'lane_change', 'merge') and c.direction != 'right' for c in claims.commitments):
        # Calculate the lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        # Graded factor for lateral execution
        if lateral_offset_change < 0:  # Ensure leftward change
            lateral_execution = 0.45 * min(1.0, abs(lateral_offset_change) / 0.5)
            lateral_nudge = 0.45 if lateral_execution > 0 else 0.0

    return {
        "perceptual_cyclist": perceptual_cyclist,
        "lateral_nudge": lateral_nudge,
        "lateral_execution": lateral_execution
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
