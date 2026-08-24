"""clip 0dd725e0-11c1-47b8-b061-e64198785267 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.24, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene with pedestrian yielding and automobile proximity.
    
    Decisive Events:
    1. Yielding to pedestrians: Expect mention of 'pedestrian' and a 'decelerate' commitment.
       Trajectory should show a speed drop of at least 2.25 m/s around t=3.3 s.
    """
    # Initialize component scores
    components = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0
    }

    # Check perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        components["mention_pedestrian"] = 0.1

    # Check commitment and trajectory for pedestrian yielding
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop, considering the timing
        min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints))
        if 3.0 <= min_speed_time <= 3.5:
            components["decelerate_for_pedestrian"] = 0.6 * min(1.0, speed_drop / 4.5)

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
