"""clip d75b8a9d-5773-4193-938e-653faf517e68 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene d75b8a9d-5773-4193-938e-653faf517e68:
    - Decelerate in response to a red traffic light ahead (speed drop of at least 0.85 m/s).
    - Mention of traffic control entities like 'signal' or 'intersection'.
    """
    comp = {
        "mention_traffic_control": 0.0,
        "decelerate_executed": 0.0,
    }

    # Perceptual mention of traffic control entities
    if any(p.entity in ('signal', 'intersection') for p in claims.perceptual):
        comp["mention_traffic_control"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration execution
        comp["decelerate_executed"] = 0.6 * min(1.0, speed_drop / 1.7)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
