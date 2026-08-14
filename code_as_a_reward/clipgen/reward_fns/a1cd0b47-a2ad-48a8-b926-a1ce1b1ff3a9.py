"""clip a1cd0b47-a2ad-48a8-b926-a1ce1b1ff3a9 - attempt 3/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring a rollout based on the scene's decisive events:
    1. Pedestrian crossing: Expect deceleration with a speed drop of at least 0.2 m/s.
    """
    # Initialize component scores
    comp = {
        "decelerate_executed": 0.0
    }

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after
        # Graded factor for deceleration execution
        comp["decelerate_executed"] = 0.9 * min(1.0, speed_drop / 0.4)

    # Check for perceptual mention of pedestrian
    if any(p.entity in ('pedestrian', 'cyclist') for p in claims.perceptual):
        comp["decelerate_executed"] += 0.1  # Additive credit for mention

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
