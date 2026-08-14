"""clip dc361cc4-4102-45cb-9b82-2a9b755a1f46 - attempt 5/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 8)"""
def components(claims, traj):
    """Components for scene dc361cc4-4102-45cb-9b82-2a9b755a1f46:
    - Decisive event: Merge into the right lane through a construction zone.
    - Perceptual mention: 'construction_cones' or 'work_zone'.
    - Commitment: Lateral maneuver ('merge', 'lane_change') to the right.
    - Trajectory: Rightward lateral offset change of at least 0.225 m.
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Perceptual mention check
    if any(p.entity in ('construction_cones', 'work_zone') for p in claims.perceptual):
        perceptual_score = 0.05  # Mention-only credit

    # Commitment check for lateral maneuver to the right
    if any(c.maneuver in ('lane_change', 'merge') and c.direction != 'left' for c in claims.commitments):
        # Trajectory check for rightward lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        if lateral_offset_change < -0.225:  # Rightward movement
            trajectory_score = min(1.0, abs(lateral_offset_change) / 0.45)
        else:
            trajectory_score = 0.0

        # Combine commitment and trajectory for lateral maneuver
        commitment_score = 0.65 * trajectory_score

    # Return component scores
    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver": commitment_score,
    }

def reward(claims, traj):
    # Calculate total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
