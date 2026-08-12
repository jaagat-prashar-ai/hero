"""clip 6303353e-7b55-4216-b1a1-64b362e76aba - attempt 2/3 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event:
    Stopping for pedestrians crossing the crosswalk.
    - Perceptual claims: Detecting pedestrians and crosswalk.
    - Commitment claims: Committing to stop.
    - Trajectory execution: Decelerating to a stop within the time window.
    """
    # Initialize component scores
    saw_pedestrian = 0.0
    saw_crosswalk = 0.0
    committed_to_stop = 0.0
    executed_stop = 0.0

    # Check perceptual claims
    if any(pc.entity == 'pedestrian' and pc.state == 'crossing' for pc in claims.perceptual):
        saw_pedestrian = 0.2

    if any(pc.entity == 'crosswalk' and pc.state == 'crossing' for pc in claims.perceptual):
        saw_crosswalk = 0.1

    # Check commitment claims
    if any(cc.maneuver == 'stop' and cc.speed_profile == 'decelerate' for cc in claims.commitments):
        committed_to_stop = 0.2

    # Check trajectory execution with conjunction of commitment
    if traj.n_waypoints > 0 and committed_to_stop > 0.0:
        # Check if the trajectory shows a significant speed drop to a stop
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        if len(speed_window) > 0:
            initial_speed = speed_window[0]
            final_speed = speed_window[-1]
            speed_drop = initial_speed - final_speed

            # Allow some noise in speed reduction
            if speed_drop >= 7.0 and final_speed <= 0.5:
                executed_stop = 0.5

    return {
        "saw_pedestrian": saw_pedestrian,
        "saw_crosswalk": saw_crosswalk,
        "committed_to_stop": committed_to_stop,
        "executed_stop": executed_stop
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
