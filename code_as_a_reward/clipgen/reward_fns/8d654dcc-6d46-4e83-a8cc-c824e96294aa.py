"""clip 8d654dcc-6d46-4e83-a8cc-c824e96294aa - attempt 1/5 - gate PASS (pos 0.70, max pert 0.21, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the scene's decisive event:
    - Steering adjustment to maintain a safe distance from cyclists on the right.
    - Expect a leftward nudge maneuver and a mention of cyclists.
    - Trajectory should show a leftward lateral offset change of at least ~1.7 m.
    """

    # Initialize component scores
    perceptual_mention = 0.0
    lateral_execution = 0.0

    # Check for perceptual mention of cyclists
    if any(p.entity in ('cyclist',) for p in claims.perceptual):
        perceptual_mention = 0.1

    # Check for lateral commitment to nudge left
    if any(c.maneuver in ('nudge', 'lane_change', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate leftward lateral offset change
        initial_offset = traj.lateral_offset_m[0]
        final_offset = traj.final_lateral_offset_m
        leftward_offset_change = initial_offset - final_offset  # Positive if moving left

        # Graded factor for lateral execution
        lateral_execution = 0.6 * min(1.0, leftward_offset_change / 3.4)

    return {
        "perceptual_mention": perceptual_mention,
        "lateral_execution": lateral_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
