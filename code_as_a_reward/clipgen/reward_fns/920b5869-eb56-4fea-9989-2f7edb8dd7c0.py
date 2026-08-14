"""clip 920b5869-eb56-4fea-9989-2f7edb8dd7c0 - attempt 3/5 - gate PASS (pos 0.79, max pert 0.10, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the scene's decisive events:
    1. Steering left to maintain a safe distance from a construction worker.
       - Perceptual mention of 'workers' or 'pedestrian'.
       - Lateral maneuver commitment to steer left (nudge, lane_change, turn).
       - Graded trajectory factor based on leftward heading change.
    """
    # Initialize component scores
    perceptual_worker = 0.0
    lateral_maneuver = 0.0

    # Check for perceptual mention of the construction worker
    if any(p.entity in ('workers', 'pedestrian') for p in claims.perceptual):
        perceptual_worker = 0.1

    # Check for a lateral maneuver commitment to steer left
    if any(c.maneuver in ('nudge', 'lane_change', 'turn') and c.direction != 'right' for c in claims.commitments):
        # Calculate the leftward heading change
        heading_change = traj.total_heading_change_deg
        if heading_change < 0:  # Negative indicates leftward
            # Graded trajectory factor based on the magnitude of the heading change
            lateral_maneuver = 0.7 * min(1.0, abs(heading_change) / 17.0)

    return {
        "perceptual_worker": perceptual_worker,
        "lateral_maneuver": lateral_maneuver,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
