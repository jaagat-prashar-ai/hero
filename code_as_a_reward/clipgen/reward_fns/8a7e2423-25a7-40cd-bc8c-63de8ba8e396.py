"""clip 8a7e2423-25a7-40cd-bc8c-63de8ba8e396 - attempt 2/5 - gate PASS (pos 0.90, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with stop sign and pedestrian crossing.
    Decisive events:
    1. Anticipated stop for stop sign and pedestrians (speed_profile='decelerate').
    Trajectory thresholds:
    - Speed drop floor: 3.1 m/s (half of 6.2 m/s drop).
    """
    # Initialize component scores
    comp = {
        "saw_pedestrian": 0.0,
        "decelerate_executed": 0.0
    }

    # Check for perceptual claims
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.1

    # Check for commitment claims and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.1, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after
        # Graded speed drop factor with timing consideration
        if min_speed_after < initial_speed:
            comp["decelerate_executed"] = 0.9 * min(1.0, speed_drop / 3.1)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
