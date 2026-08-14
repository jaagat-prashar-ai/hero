"""clip 1da15fc3-05fa-4cc1-a3c7-56e8e8df83c4 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing event.
    - Pedestrian crossing: expect mention of 'pedestrian' and potential deceleration.
    - Trajectory: graded speed change for deceleration claims.
    """
    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for deceleration commitment and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.final_speed_mps - traj.initial_speed_mps
        # Graded trajectory factor for deceleration
        deceleration_factor = 0.6 * min(1.0, speed_drop / 1.0)

        # Assign deceleration credit if pedestrian is mentioned
        if any(p.entity == 'pedestrian' for p in claims.perceptual):
            comp["decelerate_for_pedestrian"] = deceleration_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
