"""clip 67b49d46-f590-48fb-9aeb-41a68e5b135e - attempt 3/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing.
    
    Decisive event:
    1. Pedestrian crossing: Expect mention of 'pedestrian' and a deceleration
       commitment. Trajectory should show a speed drop of at least 1.0 m/s
       primarily in the latter half of the window.
    """
    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_executed": 0.0
    }

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop in the latter half of the window
        speed_series = np.array(traj.speed_mps)
        min_speed_after_half = np.min(window(speed_series, traj.dt_s, 3.0, 6.4))
        speed_drop = traj.initial_speed_mps - min_speed_after_half
        comp["decelerate_executed"] = 0.9 * min(1.0, speed_drop / 2.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
