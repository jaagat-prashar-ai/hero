"""clip d30a4a99-816a-49eb-a555-fcdc7a6fb800 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 10)"""
def components(claims, traj):
    """Components for scene d30a4a99-816a-49eb-a555-fcdc7a6fb800:
    - Decisive Event: Pedestrian Crossing
      - Perceptual mention: pedestrian
      - Commitment: decelerate (family-level match for yield)
      - Trajectory: slight deceleration expected (graded factor)
    """
    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = speed_series.min()
        speed_drop = initial_speed - min_speed_after

        # Graded factor for deceleration
        comp["decelerate_for_pedestrian"] = 0.6 * min(1.0, speed_drop / 1.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
