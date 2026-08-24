"""clip d0a0cbf0-bcc2-4c72-9a54-f542dc99f560 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Pedestrian Crossing: Expect mention of 'pedestrian' and a deceleration commitment.
       Trajectory should show a speed drop of at least 0.05 m/s early in the window.
    """
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0
    }

    # Perceptual mention of pedestrian
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Commitment to decelerate (stop/yield/wait/decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded trajectory factor for deceleration
        if speed_drop >= 0.05:
            comp["decelerate_for_pedestrian"] = 0.65 * min(1.0, speed_drop / 0.1)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
